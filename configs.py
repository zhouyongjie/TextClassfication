
class Config:
    def __init__(self):

        self.model_name = ""
        self.data_type = "TouTiao"
        self.save_model_path = "./result"

        self.train_data_path = "./data/toutiaonews38w/train.csv"
        self.dev_data_path = "./data/toutiaonews38w/dev.csv"
        self.label2idx = {'民生': 0, '文化': 1, '娱乐': 2, '体育': 3, '财经': 4, '房产': 5,
                          '汽车': 6, '教育': 7, '科技': 8, '军事': 9, '旅游': 10, '国际': 11,
                          '证券': 12, '农业': 13, '电竞': 14}
        self.num_classes = 16

        self.epoch = 10
        self.print_every_batch = 40
        self.max_length = 128
        self.batch_size = 16
        self.lr = 2e-5


config = Config()


