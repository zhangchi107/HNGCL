Heterogeneous Neighborhood-Enhanced Graph Contrastive Learning for Recommendation

We will organize the complete code and upload it after the paper is accepted for publication.
### Enviroments
- python==3.10
- pytorch==2.0
- cuda==118
- dgl==2.0
## How to Run the code
### Environment Installation
```
pip install torch==2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```
pip install  dgl==2.0 -f https://data.dgl.ai/wheels/cu118/repo.html
```
### Running on Amazon, Douban-Movie, Yelp, and Movielens Datasets
```
python main_HNGCL.py --dataset=Yelp --device='cuda:0' --lr=0.0005 --dim=128  --epochs=100  --batch_size=1024 --test_batch_size=300  --verbose=1  --GCN_layer=2 --cl_rate=0.01 --gamma=1.0 --beta=1.0
```
```
python main_HNGCL.py --dataset=DoubanBook --device='cuda:0' --lr=0.0005 --dim=128  --epochs=100  --batch_size=1024 --test_batch_size=300  --verbose=1  --GCN_layer=1 --cl_rate=0.1 --gamma=1.0 --beta=1.0
```
``` 
python main_HNGCL.py --dataset=Yelp --device='cuda:0' --lr=0.0005 --dim=128  --epochs=100  --batch_size=1024 --test_batch_size=300  --verbose=1  --GCN_layer=1 --cl_rate=0.2 --gamma=1.0 --beta=1.0
```
