import cv2
import time

start_time = time.time()

#bunch of objects created to test
brisk = cv2.BRISK_create()

mser = cv2.MSER_create()
fast = cv2.FastFeatureDetector_create()
star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
akaze = cv2.AKAZE_create()
orb =cv2.ORB_create()
surf = cv2.xfeatures2d.SURF_create()
import cv2
from Model import Detector


class MIXED(Detector.Detector):
    """This class uses the ORB feature detector to get compare a reference image with that in 'the database

    Variables:
        brisk: Global variable object of BRISK
        freak: Global variable object of FREAK
    """
    global brisk
    global freak
    global finalImage
    brisk = cv2.BRISK_create()
    freak = cv2.xfeatures2d.FREAK_create()

    def __init__(self, image, imagePath):
        """
        Constructor requires two image paths
        :param image: reference image path
        :param imagePath: image path for 'database' found locally

        Variables:
            Composite: Empty array that will store a tuple of (features, image-of-feature)
            featAndImage: Populated composite
            Composite2: Empty array that will store a tuple of (matches, image-of-matches)
            matchesAndImage: populated composite2
        """

        self.image = image
        self.imagePath = imagePath
        # image = cv2.imread(image)

        imageSlice = self.read_in_names(imagePath)
        composite = []

        featAndImages = self.feature_extractor(composite, imageSlice)
        composite2 = []

        matchesAndImages = self.matcher(image, featAndImages, composite2)

        self.finalImage = self.matcher_helper(matchesAndImages)



    def feature_extractor(self, preMatchedSlice, imageSlice):
        """
        Extracts features using the respective feature detector

        :param preMatchedSlice: a list containing a tuple of features and the image to which the features were derived
        :param imageSlice: a list of images from the 'database'
        :return: prematchedslice- fully populated
        """
        if len(imageSlice) is 0:

            return preMatchedSlice
        else:

            image = imageSlice.pop()
            kp1 = brisk.detect(image, None)
            kp1, des1 = freak.compute(image, kp1)
            preMatchedSlice.append((des1, image))

            return self.feature_extractor(preMatchedSlice, imageSlice)

    def matcher(self, image, preMatchesSlice, postMatchesSlice):
        """
        computes matches

        :param image: reference image that we are trying to find in the database
        :param preMatchesSlice: list containing the tuple (features, images)
        :param postMatchesSlice: list containing the tuple (matches, images)
        :return: postMatchesSlice fully populated
        """
        kp1 = brisk.detect(image, None)
        kp1, des1 = freak.compute(image, kp1)

        bf = cv2.BFMatcher()

        if len(preMatchesSlice) is 0:
            return postMatchesSlice
        else:
            des2 = preMatchesSlice[len(preMatchesSlice) - 1][0]
            matches = bf.knnMatch(des1, des2, k=2)

            postMatchesSlice.append((matches, preMatchesSlice[len(preMatchesSlice) - 1][1]))
            preMatchesSlice.pop()
            return self.matcher(image, preMatchesSlice, postMatchesSlice)

    # the loops are for testing and debugging. The 'solution' stores a sorted array with the first index the matched image

    def matcher_helper(self, matches):
        """
        Using a list of tuples containing matches and images, find the best matches with lowes ratio,
        then sort the new list of tuples (good features, images) in descending order and return the image in index 0

        :param matches: list of tuples containing (matches, images)
        """
        solution = []
        # solution = sorted(matches, key = lambda  tup: len(tup[0]), reverse= True)

        for m, n in matches:
            good = []
            for m1, n1 in m:
                if m1.distance < 0.75 * n1.distance:
                    good.append([m1])

            solution.append((good, n))
            # print(len(good))

        solution = sorted(solution, key=lambda tup: len(tup[0]), reverse=True)

        # self.show_stuff("solution:", solution[0][1])
        return solution[0][1]


