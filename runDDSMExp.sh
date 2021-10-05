python python_src/NewResNetMain.py --id=CRF2-GA --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=ga --dataset=ddsm
python python_src/NewXceptionNetMain.py --id=CXF2-GA --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=ga --dataset=ddsm
python python_src/NewResNetMain.py --id=CRF2-PSO --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=pso --dataset=ddsm
python python_src/NewXceptionNetMain.py --id=CXF2-PSO --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=pso --dataset=ddsm
