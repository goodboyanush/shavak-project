from skip_thoughts import configuration
from skip_thoughts import encoder_manager

# Set paths to the model.
VOCAB_FILE = "/data/anussank/shavak/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/vocab.txt"
EMBEDDING_MATRIX_FILE = "/data/anussank/shavak/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy"
CHECKPOINT_PATH = "/data/anussank/shavak/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424"
encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(), VOCAB_FILE, EMBEDDING_MATRIX_FILE, CHECKPOINT_PATH)

def getRepr(sentence):
    return encoder.encode(sentence)

