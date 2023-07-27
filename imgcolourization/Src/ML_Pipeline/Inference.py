from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage.io import imsave
import numpy as np
import math
from ML_Pipeline.References import References
# from References import References

class Inference(References):

    def processImg(self, idx, test, newmodel, model):
        """
        Processing the image and predicting the output
        :param idx:
        :param test:
        :param newmodel:
        :param model:
        :return:
        """
        test = resize(test, (224, 224), anti_aliasing=True)
        test *= 1.0 / 255
        lab = rgb2lab(test)
        l = lab[:, :, 0]
        L = gray2rgb(l)
        L = L.reshape((1, 224, 224, 3))
        vggpred = newmodel.predict(L)
        ab = model.predict(vggpred)
        ab = ab * 128
        cur = np.zeros((224, 224, 3))
        cur[:, :, 0] = l
        cur[:, :, 1:] = ab
        imsave(self.ROOT_DIR+ self.TEST_IMG+ str(idx) + ".jpg", lab2rgb(cur))

    def calculate_psnr(self, image1, image2):
        mse = np.mean((image1 - image2) ** 2)
        max_pixel = 1.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr
    
    def calculate_ssim(self, image1, image2):
        ssim_score = ssim(image1, image2, multichannel=True)
        return ssim_score