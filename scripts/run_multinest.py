import sys

from bayesian_gradabm.multinest import MultiNest

mn = MultiNest.from_file(sys.argv[1])
mn.run()
