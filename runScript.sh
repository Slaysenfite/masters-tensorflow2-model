#!/bin/bash
#
#cd $HOME/data
#
#git clone https://github.com/Slaysenfite/ddsm_lr

cd $HOME/masters-tensorflow2-model/

git checkout instance/uber

rm -r output

workon wesselsenv

#python python_src/DdsmMetadataGenerator.py

# Run Scripts
python python_src/NewUNetSegMain.py --id=SUD2-GA-2 --optimizer=sgd --dataset=cbis_seg --preloaded_experiment=SUD1 --epochs=100 --meta_heuristic=ga
python python_src/NewUNetSegMain.py --id=SUD2-PSO-2 --optimizer=sgd --dataset=cbis_seg --preloaded_experiment=SUD1 --epochs=100 --meta_heuristic=pso

python python_src/ResNetMain.py --optimizer=adam --id=CRF-I --dataset=ddsm --preloaded_weights=True --epochs=100
python python_src/NewResNetMain.py --id=CRF-PSO --optimizer=adam --preloaded_experiment=CRF-I --epochs=100 --meta_heuristic=pso --dataset=ddsm
python python_src/NewResNetMain.py --id=CRF-GA --optimizer=adam --preloaded_experiment=CRF-I --epochs=100 --meta_heuristic=ga --dataset=ddsm

python python_src/ResNetMain.py --optimizer=adam --id=CRD-I --dataset=cbis --preloaded_weights=True --epochs=100
python python_src/NewResNetMain.py --id=CRD-GA --optimizer=adam --preloaded_weights=True --meta_heuristic=ga --preloaded_experiment=CRD-I
python python_src/NewResNetMain.py --id=CRD-PSO --optimizer=adam --preloaded_weights=True --meta_heuristic=pso --preloaded_experiment=CRD-I

python python_src/XceptionNetMain.py --optimizer=adam --id=CXD1-I --dataset=cbis --preloaded_weights=True --epochs=100
python python_src/NewXceptionNetMain.py --id=CXD-GA --optimizer=adam --preloaded_weights=True --meta_heuristic=ga--preloaded_experiment=CXD1-I
python python_src/NewXceptionNetMain.py --id=CXD-PSO --optimizer=adam --preloaded_weights=True --meta_heuristic=pso --preloaded_experiment=CXD1-I



