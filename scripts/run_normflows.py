if __name__ == "__main__":
    import sys
    import torch.multiprocessing as mp
    try:
       mp.set_start_method('spawn', force=True)
       print("spawned")
    except RuntimeError:
       pass
    
    from bayesian_gradabm.normflows import NormFlows
    
    nf = NormFlows.from_file(sys.argv[1])
    nf.run()
