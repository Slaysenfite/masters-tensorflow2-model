python python_src/ResNetMain.py --optimizer=adam --id=MCRD001Pso --preloaded_experiment=MCRD001-ImageNet --dataset=cbis --epochs=100 --meta_heuristic=pso
python python_src/ResNetMain.py --optimizer=adam --id=MCRD001Ga --preloaded_experiment=MCRD001-ImageNet --dataset=cbis --epochs=100 --meta_heuristic=ga

python python_src/XceptionNetMain.py --optimizer=adam --id=MCXD001Pso --preloaded_experiment=MCXD001-ImageNet --dataset=cbis --epochs=100 --meta_heuristic=pso
python python_src/XceptionNetMain.py --optimizer=adam --id=MCXD001Ga --preloaded_experiment=MCXD001-ImageNet --dataset=cbis --epochs=100 --meta_heuristic=ga

python python_src/UNetSegMain.py --id=MSUD001Pso --preloaded_experiment=MSUD001-ImageNet --dataset=cbis_seg --optimizer=sgd --epochs=100 --meta_heuristic=pso
python python_src/UNetSegMain.py --id=MSUD001Ga --preloaded_experiment=MSUD001-ImageNet --dataset=cbis_seg --optimizer=sgd --epochs=100 --meta_heuristic=ga
