python python_src/UNetSegMain.py --id=SUD1 --dataset=cbis_seg --optimizer=sgd --preloaded_weights=False --epochs=25
python python_src/NewUNetSegMain.py --id=SUD2-GA-2 --optimizer=sgd --dataset=cbis_seg --preloaded_experiment=SUD1 --meta_heuristic=ga
python python_src/NewUNetSegMain.py --id=SUD2-PSO-2 --optimizer=sgd --dataset=cbis_seg --preloaded_experiment=SUD1 --meta_heuristic=pso


