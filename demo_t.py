from main import parse, train
from src.model_t import Tmodel

if __name__=="__main__":
    args = parse(print_help=True, model_cls=Tmodel)
    # args.n_splits = 10
    args.dataset_name = "Fdataset"
    # args.lr = 1e-3
    # args.lr = 5e-4
    # args.dropout = 0.4
    # args.disease_neighbor_num = 25
    # args.drug_neighbor_num = 25
    # args.disease_feature_topk = 20
    # args.drug_feature_topk = 20
    # args.embedding_dim = 64
    # args.neighbor_embedding_dim = 32
    # args.hidden_dims = (64, 32)
    # args.debug = True
    args.seed = 9
    # args.epochs = 1
    args.train_fill_unknown = False
    # args.comment = "test"
    # args.loss_fn = "focal"

    train(args, Tmodel)

