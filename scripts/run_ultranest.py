import sys

from bayesian_gradabm.ultranest import UltraNest

mn = UltraNest.from_file(sys.argv[1])
mn.run()
