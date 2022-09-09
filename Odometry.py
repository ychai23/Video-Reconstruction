import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import scipy.linalg
import scipy
from glob import glob
import cv2, skimage, os


class OdometryClass:
    def __init__(self, frame_path):
        """
        Params:
        
        self.frame_path: directory
        self.frames: frames of each img
        self.focal_length: focal length of camera (same for x,y)
        self.pp: cx, cy
        self.pose: ground truth of all camera positions
        self.K: intrinstic matrix of camera
        """
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
        
        
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]
            for i in range(len(self.pose)):
                self.pose[i] = [float(p) for p in self.pose[i]]
            
        self.K = [[self.focal_length, 0, self.pp[0]], [0, self.focal_length, self.pp[1]], [0, 0, 1]]
        self.P = self.K @ np.hstack((np.identity(3), np.zeros((3,1))))

        
    def decomp_essential_mat(self, K, E, q1, q2):
        """
        Decompose the Essential matrix
        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self.form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)


            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        for R, t in pairs:
            z_sum = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        R, t = right_pair

        return R, t

        
    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        """
        Get Ground truth x,y,z
        """
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])
    
    def get_sift_data(self, img):
        """
        Detect the keypoints and compute their SIFT descriptors with opencv library
        """
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        kp = np.array([k.pt for k in kp])
        return kp, des

    def get_matches(self, img1, img2):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object
        Parameters
        ----------
        i (int): The current frame
        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        
        # Find the keypoints and descriptors with ORB
        FLANN_INDEX_LSH = 6
        orb = cv2.ORB_create(3910)
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        # Find matches
        matcher = cv2.FlannBasedMatcher(indexParams=index_params,searchParams=search_params)
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        # Get the image points form the good matches
        q1 = np.array(np.float32([kp1[m.queryIdx].pt for m in good]))
        q2 = np.array(np.float32([kp2[m.trainIdx].pt for m in good]))
        
        return np.hstack((q1, q2))
        
#         print(len(matches))
        return matches
    
    def form_transf(self, R, t):
        """
        Construct transformation matrix from R and t
        """
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_pose(self, q1, q2, K, frame_id):
        """
        Calculates the transformation matrix
        """
        # Calculate essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, threshold=0.484, method=cv2.RANSAC, prob=0.9995, focal=self.focal_length, pp=self.pp)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(self.K, E, q1, q2)
        
#         # Decompose the Essential matrix into R and t
#         _, R, t, _ = cv2.recoverPose(E, q1, q2, focal=self.focal_length, pp=self.pp)
        
        # Scale t
        t = t * self.get_scale(frame_id)
        
        # Get transformation matrix
        transformation_matrix = self.form_transf(R, np.squeeze(t))
        return transformation_matrix
    
    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def plot_results(self, path):
        """
        plot results in 3D space
        """
        gt = []
        for idx in range(len(self.pose)):
            pose = self.pose[idx]
            gt.append([float(pose[3]), float(pose[7]), float(pose[11])])
        gt = np.array(gt)
        
        # plot training path
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(gt[:,0],gt[:,1],gt[:,2])
        ax.plot(path[:,0],path[:,1],path[:,2])
        plt.show()
    
    def find_matches(self, idx):
        """
        Get matches from two imgs
        """
        # get the two neighbor images
        img_left = self.imread(self.frames[idx-1])
        img_right = self.imread(self.frames[idx])

        # get matches from the two imgs
        matches = self.get_matches(img_left, img_right)
        
#         matches = self.get_best_matches(img_left, img_right, 300)
        return matches
    
    
    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """
        path = np.zeros((len(self.frames),3))
        
#         print(len(self.pose))
        for idx in range(0, len(self.frames)):
            
            if idx == 0:
                cur_pose = np.array(self.pose[0]).reshape((3,4))
            else:
                # get matches from the two imgs
                matches = self.find_matches(idx)
                
                # calculate transformation matrix
                trans_matrix = self.get_pose(matches[:,:2], matches[:,2:], self.K, idx)
                
                # tranform from previous position
                cur_pose = cur_pose @ np.linalg.inv(trans_matrix)
            
            tran = cur_pose[:3, 3]
            # assign new position to path
            path[idx] = [tran[0], tran[1], tran[2]]
#             print(path[idx])

        return path
        

if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path,path.shape)
#     odemotryc.plot_results(path)