python python_src/UNetSegMain.py --id=MSUD001 --dataset=cbis_seg --optimizer=sgd --preloaded_weights=False --epochs=100
python python_src/UNetSegMain.py --id=MSUD001-ImageNet --dataset=cbis_seg --optimizer=sgd --preloaded_weights=True --epochs=100
python python_src/UNetSegMain.py --id=MSUD001Pso --preloaded_experiment=MSUD001-ImageNet --dataset=cbis_seg --optimizer=sgd --epochs=100 --meta_heuristic=pso
python python_src/UNetSegMain.py --id=MSUD001Ga --preloaded_experiment=MSUD001-ImageNet --dataset=cbis_seg --optimizer=sgd --epochs=100 --meta_heuristic=ga
