python python_src/ResNetMain.py --id=PSO05 --epochs=1 --dataset=cbis --preloaded_weights=True --tf_fit=False --meta_heuristic=pso --meta_heuristic_layers=5
python python_src/ResNetMain.py --id=PSO10 --epochs=1 --dataset=cbis --preloaded_weights=True --tf_fit=False --meta_heuristic=pso --meta_heuristic_layers=10
python python_src/ResNetMain.py --id=PSO20 --epochs=1 --dataset=cbis --preloaded_weights=True --tf_fit=False --meta_heuristic=pso --meta_heuristic_layers=20
python python_src/ResNetMain.py --id=PSO50 --epochs=1 --dataset=cbis --preloaded_weights=True --tf_fit=False --meta_heuristic=pso --meta_heuristic_layers=50

python python_src/ResNetMain.py --id=GA05 --epochs=1 --dataset=cbis --preloaded_weights=True --tf_fit=False --meta_heuristic=ga --meta_heuristic_layers=5
python python_src/ResNetMain.py --id=GA10 --epochs=1 --dataset=cbis --preloaded_weights=True --tf_fit=False --meta_heuristic=ga --meta_heuristic_layers=10
python python_src/ResNetMain.py --id=GA20 --epochs=1 --dataset=cbis --preloaded_weights=True --tf_fit=False --meta_heuristic=ga --meta_heuristic_layers=20
python python_src/ResNetMain.py --id=GA50 --epochs=1 --dataset=cbis --preloaded_weights=True --tf_fit=False --meta_heuristic=ga --meta_heuristic_layers=50

