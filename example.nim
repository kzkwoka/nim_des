import rnim/src/rnim
import math
import random
import sequtils
import std/algorithm

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
  Individual = seq[float]
  Population = seq[Individual]
  History = seq[Population]

const
  F = sqrt(1.0 / 2.0)

var
  c = 0.1
  delta = 0.01
  epsilon = 0.001
  qmax = -Inf  
  bestValue = Inf
  bestIndividual: seq[float] = @[]

proc objectiveFunction(individual: Individual, fn_i: int, lower, upper: Individual): float =
  result = 0.0
  if (individual.allIt(it < 100)) and (individual.allIt(it > -100)):
    let fn = cint(fn_i)
    let nx = cint(individual.len)
    var input: seq[cdouble] = newSeq[cdouble](individual.len)
    for i in 0..<individual.len:
      input[i] = individual[i]
    result = cec2017(nx, fn, addr(input[0]))
    qmax = max(qmax, result)
    if result < bestValue:
      bestValue = result
      bestIndividual = individual
  else:
    var sumSquares = 0.0
    for x in individual:
      if x > 100:
        sumSquares += (x - 100) ^ 2
      if x < -100:
        sumSquares += (-100 - x) ^ 2
    result = qmax + sumSquares

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

proc sortPopulationByFitness(population: var Population, fn_i: int, lower, upper: Individual) =
  population.sort(proc (a, b: Individual): int =
    cmp(objectiveFunction(a, fn_i, lower, upper), objectiveFunction(b, fn_i, lower, upper)),
  )

proc calculateShift(previousShift, s, m: Individual): Individual =
  let dimension = s.len
  result = newSeqWith(dimension, 0.0)
  for j in 0..<dimension:
    result[j] = (1.0 - c) * previousShift[j] + c * (s[j] - m[j])

proc generateNewIndividual(s, shift: Individual, dimension: int): Individual =
  result = newSeqWith(dimension, 0.0)
  for j in 0..<dimension:
    result[j] = s[j] + shift[j] + epsilon * gauss(0.0, 1.0)

proc selectFromHistory(history: History, size, mu: int): Population =
  result = newSeq[Individual](size)
  for i in 0..<size:
    let histIndex = rand(history.len - 1)
    let popIndex = rand(mu)
    result[i] = history[histIndex][popIndex]

proc expectedValueNorm(n: int): float =
  let
    numerator = sqrt(2.0) * gamma((float(n) + 1) / 2)
    denominator = gamma(float(n) / 2)
  result = numerator / denominator

proc generateNewPopulation(history: History, populationSize, mu, dimension: int, shift: seq, s:Individual): Population = 
  var newPopulation = newSeq[Individual](populationSize)
  for i in 0..<populationSize:
    let selectedIndividuals = selectFromHistory(history, 2, mu)
    let a = selectedIndividuals[0]
    let b = selectedIndividuals[1]
    var d = newSeqWith(dimension, 0.0)
    for j in 0..<dimension:
      d[j] = F * (a[j] - b[j]) + shift[j] * delta * gauss(0.0, 1.0)
    var newIndividual = generateNewIndividual(s, d, dimension)
    newPopulation[i] = newIndividual
  result = newPopulation

proc differentialEvolutionStrategy(dimension, fn_i, maxGenerations: int): Individual =
  let populationSize = 4 * dimension
  let mu = populationSize div 2
  let H = (6 + 3 * sqrt(float(dimension)).int)
  c = 4/(dimension+4)
  delta = expectedValueNorm(dimension)
  epsilon = 1e-8/delta
  qmax = -Inf  
  bestValue = Inf
  bestIndividual = @[]

  var population = initializePopulation(populationSize, dimension)
  var shift = newSeqWith(dimension, 0.0)
  var history: History = newSeq[Population]()
  let lowerBound = newSeqWith(dimension, -100.0)
  let upperBound = newSeqWith(dimension, 100.0)
  var previousMean = newSeqWith(dimension, 0.0)

  history.add(population)
  for generation in 0..<maxGenerations:
    qmax = -Inf

    let m = calculateCenter(population, populationSize)
    sortPopulationByFitness(population, fn_i, lowerBound, upperBound)
    let s = calculateCenter(population, mu)
    shift = calculateShift(shift, s, m)

    population = generateNewPopulation(history, populationSize, mu, dimension, shift, s)

    history.add(population)
    if history.len > H:
      history.del(0)

    var stdDev = newSeqWith(dimension, 0.0)
    for j in 0..<dimension:
      for i in 0..<populationSize:
        stdDev[j] += pow((population[i][j] - previousMean[j]), 2)
      stdDev[j] = sqrt(stdDev[j] / float(populationSize - 1))
    
    previousMean = s

    var err = sum(stdDev) * 0.5
    if err < epsilon:
      break
    
  return bestIndividual

proc des*(dimension, fn_i, maxGenerations, seed: SEXP): SEXP {.exportR.} =
  let
    dimension = dimension.to(int)
    maxGenerations = maxGenerations.to(int)
    fn_i = fn_i.to(int)
    seed = seed.to(int)
  randomize(seed)
  result = nimToR(differentialEvolutionStrategy(dimension, fn_i, maxGenerations))

when isMainModule:
  let dimension = 2
  let maxGenerations = 2222
  var fn = 9
  echo fn
  var bestInd = differentialEvolutionStrategy(dimension, fn, maxGenerations)
  var lowerBound = newSeqWith(dimension, -100.0)
  var upperBound = newSeqWith(dimension, 100.0)
  echo "Best individual: ", bestInd
  echo "Best fitness: ", objectiveFunction(bestInd, fn, lowerBound, upperBound)
  echo bestValue
  echo bestInd
  fn = 3
  echo fn
  bestInd = differentialEvolutionStrategy(dimension, fn, maxGenerations)
  lowerBound = newSeqWith(dimension, -100.0)
  upperBound = newSeqWith(dimension, 100.0)
  echo "Best individual: ", bestInd
  echo "Best fitness: ", objectiveFunction(bestInd, fn, lowerBound, upperBound)
  echo bestValue
  echo bestInd
  
  # let i = 1
  # let x = @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  # let result = objectiveFunction(x, i)
  # echo result
  # let result2 = objectiveFunction(x, i)
  # echo result2


