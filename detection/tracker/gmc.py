import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import time


class GMC:
    def __init__(self, method='sparseOptFlow', downscale=2, verbose=None, calib_module=None):
        super(GMC, self).__init__()

        self.method = method
        self.downscale = max(1, int(downscale))
        self.calib_module = calib_module  # Optional: FramebyFrameCalib for camera calibration
        self.prev_rotation = None
        self.prev_position = None
        self.prev_calibration = None

        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == 'ecc':
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                                       useHarrisDetector=False, k=0.04)
            # self.gmc_file = open('GMC_results.txt', 'w')

        elif self.method == 'calib' or self.method == 'calibration':
            # Use camera calibration module
            if calib_module is None:
                raise ValueError("Error: calib_module must be provided when method='calib'")
            self.method = 'calib'

        elif self.method == 'file' or self.method == 'files':
            seqName = verbose[0]
            ablation = verbose[1]
            if ablation:
                filePath = r'tracker/GMC_files/MOT17_ablation'
            else:
                filePath = r'tracker/GMC_files/MOTChallenge'

            if '-FRCNN' in seqName:
                seqName = seqName[:-6]
            elif '-DPM' in seqName:
                seqName = seqName[:-4]
            elif '-SDP' in seqName:
                seqName = seqName[:-4]

            self.gmcFile = open(filePath + "/GMC-" + seqName + ".txt", 'r')

            if self.gmcFile is None:
                raise ValueError("Error: Unable to open GMC file in directory:" + filePath)
        elif self.method == 'none' or self.method == 'None':
            self.method = 'none'
        else:
            raise ValueError("Error: Unknown CMC method:" + method)

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False

    def update_calib(self, calib_module):
        """Update camera calibration module."""
        self.calib_module = calib_module

    def apply(self, raw_frame, detections=None):
        if self.method == 'calib':
            return self.applyCalib(raw_frame, detections)
        elif self.method == 'orb' or self.method == 'sift':
            return self.applyFeaures(raw_frame, detections)
        elif self.method == 'ecc':
            return self.applyEcc(raw_frame, detections)
        elif self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, detections)
        elif self.method == 'file':
            return self.applyFile(raw_frame, detections)
        elif self.method == 'none':
            return np.eye(2, 3)
        else:
            return np.eye(2, 3)

    def applyCalib(self, raw_frame, detections=None):
        """Apply camera motion compensation using camera calibration module."""
        H = np.eye(2, 3, dtype=np.float32)
        
        if self.calib_module is None or self.calib_module.rotation is None:
            return H
        
        current_rotation = self.calib_module.rotation
        current_position = self.calib_module.position
        current_calibration = self.calib_module.calibration
        
        if self.prev_rotation is None:
            self.prev_rotation = current_rotation.copy()
            self.prev_position = current_position.copy() if current_position is not None else None
            self.prev_calibration = current_calibration.copy()
            return H
        
        # Compute relative transformation
        R_rel = current_rotation @ self.prev_rotation.T
        
        if current_position is not None and self.prev_position is not None:
            t_rel = current_position - self.prev_position
        else:
            t_rel = np.zeros(3)
        
        # Extract 2D rotation and translation
        angle = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Scale factor
        if self.prev_calibration is not None and current_calibration is not None:
            scale = current_calibration[0, 0] / self.prev_calibration[0, 0]
        else:
            scale = 1.0
        
        # Translation in image space
        if current_calibration is not None:
            fx = current_calibration[0, 0]
            fy = current_calibration[1, 1]
            z = current_position[2] if current_position is not None and len(current_position) > 2 else 1.0
            tx = fx * t_rel[0] / max(z, 1.0)
            ty = fy * t_rel[1] / max(z, 1.0)
        else:
            tx = 0.0
            ty = 0.0
        
        # Build affine transformation matrix
        H = np.array([
            [scale * cos_a, -scale * sin_a, tx],
            [scale * sin_a,  scale * cos_a, ty]
        ], dtype=np.float32)
        
        # Update previous state
        self.prev_rotation = current_rotation.copy()
        self.prev_position = current_position.copy() if current_position is not None else None
        self.prev_calibration = current_calibration.copy()
        
        return H

    def applyEcc(self, raw_frame, detections=None):

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Run the ECC algorithm. The results are stored in warp_matrix.
        # (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria)
        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except:
            print('Warning: find transform failed. Set warp as identity')

        return H

    def applyFeaures(self, raw_frame, detections=None):

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # find the keypoints
        mask = np.zeros_like(frame)
        # mask[int(0.05 * height): int(0.95 * height), int(0.05 * width): int(0.95 * width)] = 255
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        keypoints = self.detector.detect(frame, mask)

        # compute the descriptors
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Match descriptors.
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # Filtered matches based on smallest spatial distance
        matches = []
        spatialDistances = []

        maxSpatialDistance = 0.25 * np.array([width, height])

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            return H

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatialDistance = (prevKeyPointLocation[0] - currKeyPointLocation[0],
                                   prevKeyPointLocation[1] - currKeyPointLocation[1])

                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and \
                        (np.abs(spatialDistance[1]) < maxSpatialDistance[1]):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)

        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)

        inliesrs = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances

        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliesrs[i, 0] and inliesrs[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Draw the keypoint matches on the output image
        if 0:
            matches_img = np.hstack((self.prevFrame, frame))
            matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
            W = np.size(self.prevFrame, 1)
            for m in goodMatches:
                prev_pt = np.array(self.prevKeyPoints[m.queryIdx].pt, dtype=np.int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
                curr_pt[0] += W
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))

                matches_img = cv2.line(matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
                matches_img = cv2.circle(matches_img, prev_pt, 2, tuple(color), -1)
                matches_img = cv2.circle(matches_img, curr_pt, 2, tuple(color), -1)

            plt.figure()
            plt.imshow(matches_img)
            plt.show()

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def applySparseOptFlow(self, raw_frame, detections=None):

        t0 = time.time()

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # Build mask: use only background pixels (exclude detected player/object regions).
        # People move independently of the camera, so features on their bodies
        # create incorrect motion vectors that degrade the warp estimate.
        h, w = frame.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask[:int(0.02 * h), :]  = 0   # top border
        mask[int(0.98 * h):, :]  = 0   # bottom border
        mask[:, :int(0.02 * w)]  = 0   # left border
        mask[:, int(0.98 * w):]  = 0   # right border
        if detections is not None and len(detections) > 0:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                x1 = int(np.clip(tlbr[0], 0, w - 1))
                y1 = int(np.clip(tlbr[1], 0, h - 1))
                x2 = int(np.clip(tlbr[2], 0, w - 1))
                y2 = int(np.clip(tlbr[3], 0, h - 1))
                mask[y1:y2, x1:x2] = 0

        #cv2.imwrite("E:/ttt0.bmp", frame)
        #cv2.imwrite("E:/ttt1.bmp", mask)
        # find the keypoints (background only)
        keypoints = cv2.goodFeaturesToTrack(frame, mask=mask, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Filter prevKeyPoints: remove any point that now falls inside a detection box.
        # A player may have walked over a previously-background keypoint position,
        # causing LK to follow the player's texture instead of the camera motion.
        prev_kp = self.prevKeyPoints
        if (detections is not None and len(detections) > 0
                and prev_kp is not None and len(prev_kp) > 0):
            pts = prev_kp.reshape(-1, 2)                          # (N, 2)
            valid = np.ones(len(pts), dtype=bool)
            for det in detections:
                tlbr = det[:4] / self.downscale
                in_box = ((pts[:, 0] >= tlbr[0]) & (pts[:, 0] <= tlbr[2]) &
                          (pts[:, 1] >= tlbr[1]) & (pts[:, 1] <= tlbr[3]))
                valid &= ~in_box
            if valid.sum() > 4:                                   # keep if enough remain
                prev_kp = prev_kp[valid]

        # find correspondences
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, prev_kp, None)

        # leave good correspondences only
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(prev_kp[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        t1 = time.time()

        # gmc_line = str(1000 * (t1 - t0)) + "\t" + str(H[0, 0]) + "\t" + str(H[0, 1]) + "\t" + str(
        #     H[0, 2]) + "\t" + str(H[1, 0]) + "\t" + str(H[1, 1]) + "\t" + str(H[1, 2]) + "\n"
        # self.gmc_file.write(gmc_line)

        return H

    def applyFile(self, raw_frame, detections=None):
        line = self.gmcFile.readline()
        tokens = line.split("\t")
        H = np.eye(2, 3, dtype=np.float32)
        H[0, 0] = float(tokens[1])
        H[0, 1] = float(tokens[2])
        H[0, 2] = float(tokens[3])
        H[1, 0] = float(tokens[4])
        H[1, 1] = float(tokens[5])
        H[1, 2] = float(tokens[6])

        return H

