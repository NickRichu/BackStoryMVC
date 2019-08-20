import cv2
from Model import Detector


class ORB(Detector.Detector):
    """This class uses the ORB feature detector to get compare a reference image with that in 'the database

    Variables:
        orb: Global variable object of ORB
    """
    global orb
    global finalImage
    # nFeatures = 10000
    # scaleFactor = 1.01 # Closer to 1 = more features
    # nLevels = 24 # Higher = more features
    orb = cv2.ORB_create()

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
        image = cv2.imread(image)

        imageSlice = self.read_in_names(imagePath)
        composite = []

        featAndImages = self.feature_extractor(composite, imageSlice)
        composite2 = []

        matchesAndImages = self.matcher(image, featAndImages, composite2)

        self.finalImage = self.matcher_helper(matchesAndImages)




    def feature_extractor(self, preMatchedSlice, imageSlice):
        surf = cv2.xfeatures2d.SURF_create()
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
            kp1= orb.detect(image, None)
            kp1, des1 = surf.compute(image, kp1)
            preMatchedSlice.append((des1, image))

            return self.feature_extractor(preMatchedSlice, imageSlice)

    def matcher(self, image, preMatchesSlice, postMatchesSlice):
        surf = cv2.xfeatures2d.SURF_create()
        """
        computes matches

        :param image: reference image that we are trying to find in the database
        :param preMatchesSlice: list containing the tuple (features, images)
        :param postMatchesSlice: list containing the tuple (matches, images)
        :return: postMatchesSlice fully populated
        """
        kp1 = orb.detect(image, None)
        kp1, des1 = surf.compute(image, kp1)

        bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck = True)

        if len(preMatchesSlice) is 0:
            return postMatchesSlice
        else:
            des2 = preMatchesSlice[len(preMatchesSlice) - 1][0]
            matches = bf.knnMatch(des1, des2, k=2)
            # matches = bf.match(des1,des2)
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
            print(len(good))
        #
        solution = sorted(solution, key=lambda tup: len(tup[0]), reverse=True)

        return solution[0][1]


