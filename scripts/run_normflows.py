import sys

from bayesian_gradabm.normflows import NormFlows

nf = NormFlows.from_file(sys.argv[1])
nf.run()
