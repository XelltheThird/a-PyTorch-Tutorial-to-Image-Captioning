from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='dataset_flickr8k.json',
                       image_folder='Flickr_Data/Images',
                       captions_per_image=1,
                       min_word_freq=5,
                       output_folder='results/',
                       max_len=50)

#/home/fs/NeuralNetworks/PytorchImageCapt/a-PyTorch-Tutorial-to-Image-Captioning/
