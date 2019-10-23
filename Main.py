'''COIN STACK DETECTER USING OPENCV
	Author : Sai Kishore Swaminathan
	Date : 23/10/2019'''
import cv2
import numpy as np 
import os
from kde import kde 
import matplotlib.pyplot as plt 
import glob 


class Solution:
	def __init__(self,path):
		self.myPath = path

	def resize_img(self,oriimg):
		W = 1000.
		height, width, depth = oriimg.shape
		imgScale = W/(width*3)
		newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
		return cv2.resize(oriimg,(int(newX),int(newY)))

	def make_clustered_array(self,center_points, x_axis_clusters ):
		line_clusters = []
		relevant_clusters = np.array([i for i in x_axis_clusters if len(i)>1])
		if len(relevant_clusters) !=0 :
			for i in relevant_clusters:
				selected = [j for j in center_points if j[0] in i]
				line_clusters.append(selected)
		return line_clusters

	def find_Lines(self):
		No_of_images = len(glob.glob1(self.myPath,"*.jpg"))
		numbers = np.arange(1,No_of_images+1)

		for number in numbers:
			oriimg = cv2.imread(self.myPath+"\Coins_img ({}).jpg".format(number))
			newimg = self.resize_img(oriimg)
			# print (newimg.shape)
			gray = cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
			# gray = cv2.medianBlur(gray,(5,5))
			gray = cv2.bilateralFilter(gray,13,60,60)
			output = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

			Circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,
										1,40,param1= 90,param2= 30,
										minRadius=0,maxRadius=0)

			if Circles is None:
				cv2.imshow("Vertical Lines", output)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			if Circles is not None:
				Detected = np.uint16(np.around(Circles))
				centers = []
				for x,y,r in Detected[0,:]:
					cv2.circle(output,(x,y),r,(0,255,255),3)
					cv2.circle(output,(x,y),2,(255,0,0),2)
					centers.append((x,y))
				
				X_axis_points = [i[0] for i in centers]
				clusters = kde(X_axis_points,bdw=3.5) 
				#If the model isn't being accurate try tweaking the bdw 
				
				line_clusters = self.make_clustered_array(centers,clusters)

				for i in line_clusters:
					i = np.array(i)
					X_1 = np.argmin(i[:,1])
					X_2 = np.argmax(i[:,1])
					cv2.line(output,tuple(i[X_1]),tuple(i[X_2]),(0,0,0),3)

				cv2.imshow("Vertical Lines", output)
				plt.show()
				cv2.waitKey(0)
				cv2.destroyAllWindows()
		return None 


if __name__ == '__main__':
	'''PLEASE ONLY USE PATHS TO IMAGE REPOSITORIES'''

	# myPath = os.getcwd()+"\COIN_IMAGES_NORMAL"
	myPath = os.getcwd()+"\COIN_IMAGES_RARE"
	Obj1 = Solution(myPath)
	Obj1.find_Lines()

