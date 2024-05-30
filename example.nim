import rnim/src/rnim
import math
import random

type
  PopElement = seq[float]

proc generatePopulation(popSize: int, dim: int): seq[PopElement] =
  result = newSeq[PopElement](popSize)
  for i in 0..<popSize:
    var popMember = newSeq[float](dim)
    for j in 0..<dim:
      popMember[j] = rand(-100.0 .. 100.0)
    result[i] = popMember

proc desRun(popSize: int = 1000, dim: int = 2): int =  
  const 
    F: float64 = 1.0 / sqrt(2.0)
  let
    lambda = 4*popSize
    mu = lambda/2
    population = generatePopulation(popSize, dim)
  result = 1

proc des*(popSize, dim: SEXP): SEXP {.exportR.} =
  let
    popSize = popSize.to(int)
    dim = dim.to(int)
  result = nimToR(desRun(popSize, dim))