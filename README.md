# Generate Query

## 1. Environment Configuration

tensorflow1.4, python2.7, gensim2.3.0, jieba0.39, progressbar2

## 2. File structure

- `utils.py`: data process and evaluate module
- `train.py`: train and test
- `data/`: dataset and pre-trained word embeddings folder. **Specially, you can download our dataset and pre-trained word embeddings from** [here](https://drive.google.com/open?id=1HKt6pIc6iF2J9EeRQSt5uI9MUL2NjqeT)
- `models/`: models folder

## 3. Train Model

python train.py --model_name New_Pointer_Generator
