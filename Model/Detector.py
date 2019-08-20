import abc
import cv2
import glob, os


class Detector (abc.ABC):
    def read_in_names(self, filepath):
        """
      Reads in all the names found in the filepath and calls read_in_pieces

      :param filepath: the filepath for the 'database' or images of interest
      :return: image- all images in the 'database'

      Variables:
          images: an empty array that will store the images
          imageNames: stores the names of the image to be read in

              """
        os.chdir(filepath)
        imageNames = glob.glob("*.jpg")
        images = []
        images = self.read_in_pieces(imageNames, images)
        # modified = []
        # images = self.resize_all_images(images,modified)
        return images

    # def resize_all_images(self,originals, modified):
    #     if len(originals) is 0:
    #         return modified
    #     else:
    #         modified.append(cv2.resize(originals.pop(),(360,480)))
    #         return self.read_in_pieces(originals, modified)

    def read_in_pieces(self, names, imageSlice):
        """

        :param names: an array containing the names of the files in the 'database'
        :param imageSlice: an empty array that becomes populated with images from the 'database
        :return: imageslice- fully populated
        """

        if len(names) is 0:
            return imageSlice
        else:
            imageSlice.append(cv2.imread(names.pop()))
            return self.read_in_pieces(names, imageSlice)

    def show_stuff(self, someText, someImage):
        """
        Shows the image

        :param someText: Window title
        :param someImage: variable containing the image
        """
        cv2.imshow(someText, someImage)
        cv2.waitKey(0)

    @abc.abstractmethod
    def feature_extractor(self,preMatchedSlice,imageSlice):
        pass

    @abc.abstractmethod
    def matcher(self,image, preMatchesSlice, postMatchesSlice):
        pass

    @abc.abstractmethod
    def matcher_helper(self,matches):
        pass




