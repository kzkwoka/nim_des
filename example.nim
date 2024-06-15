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

# proc cec2017(nx: cint, fn: cint, input: ptr): cdouble {.importc: "cec2017".}
proc cec2017(nx: cint, fn: cint, input: ptr cdouble): cdouble {.importc: "cec2017".}

type
  Individual = seq[float]
  Population = seq[Individual]

const
  F = sqrt(1.0 / 2.0)
  c = 0.1
  delta = 0.01
  epsilon = 0.001

var maxValue = -Inf
var bestValue = Inf

proc objectiveFunction(individual: Individual, fn_i: int): float =
  result = 0.0
  if (individual.allIt(it < 100)) and (individual.allIt(it > -100)):
    let fn = cint(fn_i)
    let nx = cint(individual.len)
    var input: seq[cdouble] = newSeq[cdouble](individual.len)
    for i in 0..<individual.len:
      input[i] = individual[i]
    result = cec2017(nx, fn, addr(input[0]))  # Pass the address of the first element
    maxValue = max(maxValue, result)
    bestValue = min(bestValue, result)
  else:
    var sumSquares = 0.0
    for x in individual:
      if x > 100:
        sumSquares += (x - 100) ^ 2
      if x < -100:
        sumSquares += (100 - x) ^ 2
    result = maxValue + sumSquares
  # result = float(result)

proc initializePopulation(populationSize, dimension: int): Population =
  result = newSeq[Individual](populationSize)
  for i in 0..<populationSize:
    result[i] = newSeqWith(dimension, rand(-100.0..100.0))

proc calculateCenter(population: Population, size: int): Individual =
  let dimension = population[0].len
  result = newSeqWith(dimension, 0.0)
  for i in 0..<size:
    for j in 0..<dimension:
      result[j] += population[i][j]
  for j in 0..<dimension:
    result[j] /= float(size)

proc sortPopulationByFitness(population: var Population, fn_i: int) =
  population.sort(proc (a, b: Individual): int =
    cmp(objectiveFunction(a, fn_i), objectiveFunction(b, fn_i))
  )

proc calculateShift(previousShift, s, m: Individual): Individual =
  let dimension = s.len
  result = newSeqWith(dimension, 0.0)
  for j in 0..<dimension:
    result[j] = (1.0 - c) * previousShift[j] + c * (s[j] - m[j])

proc randomNormal(mu, sigma: float): float =
  let u1 = rand(1.0)
  let u2 = rand(1.0)
  result = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2) * sigma + mu  # Adjusted log base to natural

proc generateNewIndividual(s, shift: Individual, dimension: int): Individual =
  result = newSeqWith(dimension, 0.0)
  for j in 0..<dimension:
    result[j] = s[j] + shift[j] + epsilon * randomNormal(0.0, 1.0)

proc differentialEvolutionStrategy(dimension, fn_i, maxGenerations: int): Individual =
  let populationSize = 4 * dimension
  let mu = populationSize div 2
  var population = initializePopulation(populationSize, dimension)
  var shift = newSeqWith(dimension, 0.0)

  for generation in 0..<maxGenerations:
    maxValue = -Inf
    let m = calculateCenter(population, populationSize)
    sortPopulationByFitness(population, fn_i)
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
        stdDev[j] += pow((population[i][j] - s[j]), 2)
      stdDev[j] = sqrt(stdDev[j] / float(populationSize - 1))
    var err = sum(stdDev) * 0.5
    if err < epsilon:
      break
    # if stdDev.allIt(it < epsilon):
    #   break

  return population[0]

proc des*(dimension, fn_i, maxGenerations: SEXP): SEXP {.exportR.} =
  let
    dimension = dimension.to(int)
    maxGenerations = maxGenerations.to(int)
    fn_i = fn_i.to(int)
  result = nimToR(differentialEvolutionStrategy(dimension, fn_i, maxGenerations))

when isMainModule:
  let dimension = 2
  let maxGenerations = 10000
  let bestIndividual = differentialEvolutionStrategy(dimension, 1, maxGenerations)
  echo "Best individual: ", bestIndividual
  echo "Best fitness: ", objectiveFunction(bestIndividual, 1)
  
  # let i = 1
  # let x = @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  # let result = objectiveFunction(x, i)
  # echo result
  # let result2 = objectiveFunction(x, i)
  # echo result2


