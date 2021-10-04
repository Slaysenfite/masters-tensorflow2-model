python python_src/NewResNetMain.py --id=CRD2-GA --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=ga
python python_src/NewXceptionNetMain.py --id=CXD2-GA --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=ga
python python_src/UNetSegMainNew.py --id=SUD2-GA --optimizer=sgd --dataset=cbis_seg --preloaded_weights=True --epochs=100 --meta_heuristic=ga
python python_src/NewResNetMain.py --id=CRD2-PSO --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=pso
python python_src/NewXceptionNetMain.py --id=CXD2-PSO --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=pso
python python_src/UNetSegMainNew.py --id=SUD2-PSO --optimizer=sgd --dataset=cbis_seg --preloaded_weights=True --epochs=100 --meta_heuristic=pso