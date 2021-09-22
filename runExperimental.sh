python python_src/NewResNetMain.py --id=MC2RD001 --optimizer=adam ---preloaded_weights=True --epochs=100
python python_src/NewXceptionNetMain.py --id=MC2XD001 --optimizer=adam ---preloaded_weights=True --epochs=100
python python_src/UNetSegMainNew.py --id=MS2UD001 --optimizer=sgd --dataset=cbis_seg --preloaded_weights=True --epochs=100