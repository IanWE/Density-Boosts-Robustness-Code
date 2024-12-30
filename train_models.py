from core import models



if __name__ == '__main__':
    models.get_base_nn()        #Train vanilla EMBER NN
    models.get_base_lightgbm()  #Train LightGBM
    models.get_ltnn()           #Train LT NN
    models.get_binarized_model()#Train Binarized NN
    #you can change the ratio to train models with other densities.
    models.get_compressed_model(binarization='histogram',tag='2017',ratio=16)#Train SC NN, No binarization/density 8/No density boosting
    models.get_compressed_model(binarization=False,tag='2017',ratio=8)#Train SC NN, No binarization/density 8/No density boosting
    models.get_compressed_model(binarization='bundle',tag='2017',ratio=8)#Train SCB NN, Binarization/Density 8/No density boosting
    models.get_compressed_density_model(binarization='bundle',tag='2017',ratio=16,d="density0")#Train SCBDB NN, Binarization/Density 8/Density Boosting. 0 means training with varied perturbation
    #Please refer to original PAD code repository for training of PAD models. (we also shared the modified code used for training PAD models with our strategies. Please refer to modified/ directory.).
