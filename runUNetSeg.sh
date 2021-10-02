python python_src/UNetSegMain.py --id=MSUD001 --dataset=cbis_seg --optimizer=sgd --preloaded_weights=False --epochs=50
python python_src/UNetSegMain.py --id=MSUD001-ImageNet --dataset=cbis_seg --optimizer=sgd --preloaded_weights=True --epochs=50
python python_src/UNetSegMain.py --id=MSUD001Pso --preloaded_experiment=MSUD001-ImageNet --dataset=cbis_seg --optimizer=sgd --epochs=50 --meta_heuristic=pso --preloaded_weights=True
python python_src/UNetSegMain.py --id=MSUD001Ga --preloaded_experiment=MSUD001-ImageNet --dataset=cbis_seg --optimizer=sgd --epochs=50 --meta_heuristic=ga --preloaded_weights=True
