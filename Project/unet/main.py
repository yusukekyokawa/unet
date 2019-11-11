from model import *
from data import *
import glob

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)


if __name__ == "__main__":
    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    model = unet("unet_membrane.hdf5")
    # model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    # model.fit_generator(myGene,steps_per_epoch=300,epochs=100,callbacks=[model_checkpoint])
    for folder in glob.glob("../ARC_DATAS_CROP/*"):
        num = len(glob.glob(folder + "/*.jpg"))
        testGene = testGenerator(folder)
        results = model.predict_generator(testGene, num, verbose=1)
        saveResult(folder,results)