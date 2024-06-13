import rnim/src/rnim
import math
import random
import sequtils
import std/algorithm


# {.compile: "cec2017/src/cec2017.c".}
{.compile: "cec/cec2017nim.c".}
{.compile: "cec/src/affine_trans.c".}
{.compile: "cec/src/basic_funcs.c".}
{.compile: "cec/src/cec.c".}
{.compile: "cec/src/complex_funcs.c".}
{.compile: "cec/src/hybrid_funcs.c".}
{.compile: "cec/src/interfaces.c".}
{.compile: "cec/src/utils.c".}

proc cec2017(nx: cint, fn: cint, input: ptr): cdouble {.importc: "cec2017".}


type
  # Individual = array[1000, float]
  Individual = seq[float]
  Population = seq[Individual]

const
  F = sqrt(1.0 / 2.0)
  c = 0.1
  delta = 0.01
  epsilon = 0.001

proc objectiveFunction(individual: Individual): float =
# Implementacja funkcji celu, np. Sphere Function
  result = 0.0
  let fn = cint(1)
  let nx = cint(individual.len)
  var input: ptr UncheckedArray[cdouble] = cast[ptr UncheckedArray[cdouble]](alloc0(individual.len * sizeof(cdouble)))
  for i in 0..<individual.len:
    input[i] = individual[i]
  result = cec2017(nx, fn, input)
  # result = float(result)
  dealloc(input)
    # result = 0.0
  # for x in individual:
  #   result += x * x

proc initializePopulation(populationSize, dimension: int): Population =
  result = newSeq[Individual](populationSize)
  for i in 0..<populationSize:
    result[i] = newSeqWith(dimension, rand(1.0))

proc calculateCenter(population: Population, size: int): Individual =
  let dimension = population[0].len
  result = newSeqWith(dimension, 0.0)
  for i in 0..<size:
    for j in 0..<dimension:
      result[j] += population[i][j]
  for j in 0..<dimension:
    result[j] /= float(size)

proc sortPopulationByFitness(population: var Population) =
  population.sort(proc (a, b: Individual): int =
    cmp(objectiveFunction(a), objectiveFunction(b))
  )

proc calculateShift(previousShift, s, m: Individual): Individual =
  let dimension = s.len
  result = newSeqWith(dimension, 0.0)
  for j in 0..<dimension:
    result[j] = (1.0 - c) * previousShift[j] + c * (s[j] - m[j])

proc randomNormal(mu, sigma: float): float =
  # Implementacja generowania liczby z rozkładu normalnego
  let u1 = rand(1.0)
  let u2 = rand(1.0)
  result = sqrt(-2.0 * log(u1, 10)) * cos(2.0 * PI * u2) * sigma + mu
#TODO: check podstawe algorytmu

proc generateNewIndividual(s, shift: Individual, dimension: int): Individual =
  result = newSeqWith(dimension, 0.0)
  for j in 0..<dimension:
    result[j] = s[j] + shift[j] + epsilon * randomNormal(0.0, 1.0)

proc differentialEvolutionStrategy(dimension, maxGenerations: int): Individual =
  let populationSize = 4 * dimension
  let mu = populationSize div 2
  var population = initializePopulation(populationSize, dimension)
  var shift = newSeqWith(dimension, 0.0)

  for generation in 0..<maxGenerations:
    let m = calculateCenter(population, populationSize)
    sortPopulationByFitness(population)
    let s = calculateCenter(population, mu)
    shift = calculateShift(shift, s, m)

    var newPopulation = newSeq[Individual](populationSize)
    for i in 0..<populationSize:
      let a = population[rand(populationSize - 1)]
      let b = population[rand(populationSize - 1)]
      var d = newSeqWith(dimension, 0.0)
      for j in 0..<dimension:
        d[j] = F * (a[j] - b[j]) + shift[j] * delta * randomNormal(0.0, 1.0)
      newPopulation[i] = generateNewIndividual(s, d, dimension)
    population = newPopulation

    # Warunek stopu
    var stdDev = newSeqWith(dimension, 0.0)
    for j in 0..<dimension:
      for i in 0..<populationSize:
        stdDev[j] += pow((population[i][j] - s[j]),2)
      stdDev[j] = sqrt(stdDev[j] / float(populationSize - 1))
    if stdDev.allIt(it < epsilon):
      break

  return population[0]

proc des*(dimension, maxGenerations: SEXP): SEXP {.exportR.} =
  let
    dimension = dimension.to(int)
    maxGenerations = maxGenerations.to(int)
  result = nimToR(differentialEvolutionStrategy(dimension, maxGenerations))

when isMainModule:
  let dimension = 10
  let maxGenerations = 10000
  let bestIndividual = differentialEvolutionStrategy(dimension, maxGenerations)
  echo "Best individual: ", bestIndividual
  echo "Best fitness: ", objectiveFunction(bestIndividual)
  
  # let i = 1
  # let x = @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  # let result = objectiveFunction(x)
  # echo result
  # let result2 = objectiveFunction(x)
  # echo result2


