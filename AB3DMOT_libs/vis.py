import numpy as np, cv2, random
from PIL import Image
from AB3DMOT_libs.box import Box3D
from xinshuo_visualization import random_colors
import open3d as o3d
from matplotlib import pyplot as plt 
random.seed(0)
max_color = 30
colors = random_colors(max_color)       # Generate random colors
"""The following code is takend from the nuscenes-devkit"""

import copy
import os.path as osp
import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict
from matplotlib.axes import Axes
from pyquaternion import Quaternion

def draw_box3d_image(image, qs, img_size=(900, 1600), color=(255,255,255), thickness=4):
	''' Draw 3d bounding box in image
	    qs: (8,2) array of vertices for the 3d box in following order:
	        1 -------- 0
	       /|         /|
	      2 -------- 3 .
	      | |        | |
	      . 5 -------- 4
	      |/         |/
	      6 -------- 7
	'''

	def check_outside_image(x, y, height, width):
		if x < 0 or x >= width: return True
		if y < 0 or y >= height: return True

	# if 6 points of the box are outside the image, then do not draw
	pts_outside = 0
	for index in range(8):
		check = check_outside_image(qs[index, 0], qs[index, 1], img_size[0], img_size[1])
		if check: pts_outside += 1
	if pts_outside >= 6: return image, False

	# actually draw
	if qs is not None:
		qs = qs.astype(np.int32)
		for k in range(0,4):
			i,j=k,(k+1)%4
			image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

			i,j=k+4,(k+1)%4 + 4
			image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

			i,j=k,k+4
			image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

	return image, True

proj_velo2cam2 = [[-0.999924, 0.00541796, -0.01104782, 0.04045591], 
                  [0.0119829, 0.22472467, -0.97434861, 5.42585011], 
                  [-0.00279627, -0.97440715, -0.22477256, 1.281914158]]
proj_velo2cam2 = np.array(proj_velo2cam2)

p2 = [[1286.301970, 0.000000, 986.714019],
[0.000000, 1285.731069 ,608.462901],
[0.000000, 0.000000 ,1.000000]]
p2 = np.array(p2)
def vis_obj(box, img, calib, hw, color_tmp=None, str_vis=None, thickness=4, id_hl=None, err_type=None):
	# visualize an individual object	
	# repeat is for highlighted objects, used to create pause in the video

	# draw box

	x,y,z = box.x,box.y,box.z
	h,w,l = box.h,box.w,box.l
	rot = box.ry
	# x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
	#z_corners = [0, 0, 0, 0, h, h, h, h]
	# z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
	x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
	y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
	z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
	corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
	R = np.array([[np.cos(rot), -np.sin(rot), 0],
                    [np.sin(rot), np.cos(rot), 0],
                    [0, 0, 1]])
	corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])
	corners_3d_hom = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
	corners_img = np.matmul(corners_3d_hom, proj_velo2cam2.T)
	corners_img = np.matmul(corners_img, p2.T)
	obj_pts_2d = corners_img[:, :2] / corners_img[:, 2][:, None]
	img, draw = draw_box3d_image(img, obj_pts_2d, hw, color=color_tmp, thickness=thickness)
	# draw text
	if draw and obj_pts_2d is not None and str_vis is not None:
		x1, y1 = int(obj_pts_2d[4, 0]), int(obj_pts_2d[4, 1])
		img = cv2.putText(img, str_vis, (x1+5, y1-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_tmp, int(thickness/2))

	# highlight
	if err_type is not None:
		
		# compute the radius of the highlight
		xmin = np.min(obj_pts_2d[:, 0]); xmax = np.max(obj_pts_2d[:, 0])
		ymin = np.min(obj_pts_2d[:, 1]); ymax = np.max(obj_pts_2d[:, 1])
		radius = int(max(ymax - ymin, xmax - xmin) / 2 * 1.5)
		radius = max(radius, 50)

		# draw highlighting circle
		center = np.average(obj_pts_2d, axis=0)
		center = tuple(center.astype('int16'))
		img = cv2.circle(img, center, radius, (255, 0, 0), 4)		

		# draw error message
		pos_x, pos_y = center[0] - radius, center[1] - radius - 10
		font = cv2.FONT_HERSHEY_TRIPLEX
		font_scale = 1
		font_thickness = 2
		text_size, _ = cv2.getTextSize(err_type, font, font_scale, font_thickness)
		text_w, text_h = text_size
		cv2.rectangle(img, (pos_x, pos_y - text_h - 5), (pos_x + text_w, pos_y + 5), (255, 255, 255), -1) 		# add white background
		img = cv2.putText(img, err_type, (pos_x, pos_y), font, font_scale, (255, 0, 0), font_thickness) 

	return img

def vis_image_with_obj(img, obj_res, obj_gt, calib, hw, save_path, h_thres=0, \
	color_type='det', id_hl=None, repeat=60):
	# obj_res, obj_gt, a list of object3D class instances
	# h_thres: height threshold for filtering objects
	# id_hl: ID to be highlighted, color_type: ['det', 'trk'], trk means different color for each one
	# det means th same color for the same object

	# load image
	img = np.array(Image.open(img))

	# loop through every objects
	for obj in obj_res:
		depth = obj.z
		#if depth >= 2: 		# check in front of camera
		if True:

			# obtain object color and thickness
			if color_type == 'trk':   
				color_id = obj.id 		# vary across objects
				thickness = 5
			elif color_type == 'det': 
				if id_hl is not None and obj.id in id_hl:
					# color_id = 29 		# fixed, red for highlighted errors
					color_id = obj.id * 9 	# some magic number to scale up the id so that nearby ID does not have similar color
					thickness = 5			# highlighted objects are thicker
				else:						
					color_id = 13			# general object, fixed, lightgreen
					thickness = 1			# general objects are thin
			color_tmp = tuple([int(tmp * 255) for tmp in colors[color_id % max_color]])

			# get miscellaneous information
			box_tmp = obj.get_box3D()
			str_vis = 'ID: %d' % obj.id
			
			# retrieve index in the id_hl dict
			if id_hl is not None and obj.id in id_hl:
				err_type = id_hl[obj.id]
			else:
				err_type = None
			img = vis_obj(box_tmp, img, calib, hw['image'], color_tmp, str_vis, thickness, id_hl, err_type)

	# save image
	img = Image.fromarray(img)
	img = img.resize((hw['image'][1], hw['image'][0]))
	img.save(save_path)

	# create copy of the same image with highlighted objects to pause
	if id_hl is not None:
		for repeat_ in range(repeat):
			save_path_tmp = save_path[:-4] + '_repeat_%d' % repeat_ + save_path[-4:]
			img.save(save_path_tmp)

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points   

def vis_lidar_with_obj(lid, obj_res, obj_gt, calib, hw, save_path, h_thres=0, \
	color_type='det', id_hl=None, repeat=60):
	# obj_res, obj_gt, a list of object3D class instances
	# h_thres: height threshold for filtering objects
	# id_hl: ID to be highlighted, color_type: ['det', 'trk'], trk means different color for each one
	# det means th same color for the same object

	# load image
	pcd = o3d.io.read_point_cloud(lid)
	points = np.asarray(pcd.points)
	points = points.T
	_, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=200)
	points = view_points(points[:3, :], np.eye(4), normalize=False)

	dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
	#colors = np.minimum(1, dists / 35)
	colors = np.zeros(dists.shape)
	#colors = np.full(dists.shape,255)
	ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

	# boxes_est = _second_det_to_nusc_box(det)
 
	# # Show EST boxes.
	# for box in boxes_est:
	#     if box.score >= conf_th:
	#         box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

	box_list = []
	# loop through every objects
	for obj in obj_res:
		# get miscellaneous information
		box_tmp = obj.get_box3D()
		x,y,z,h,w,l,rot,s,_ = box_tmp.__dict__.values()
		box3d = np.array([x,y,z,w,l,h,0,0,rot])
		str_vis = 'ID: %d' % obj.id
		labels=obj.id
		scores = s
		quat = Quaternion(axis=[0, 0, 1], radians=box3d[-1])
		velocity = (*box3d[6:8], 0.0)
		box = Box(
			list(box3d[:3]),
			list(box3d[3:6]),
			quat,
			label=labels,
			score=scores,
			velocity=velocity,
		)
		box_list.append(box)
		# # retrieve index in the id_hl dict
		# if id_hl is not None and obj.id in id_hl:
		# 	err_type = id_hl[obj.id]
		# else:
		# 	err_type = None
		# img = vis_obj(box_tmp, img, calib, hw['image'], color_tmp, str_vis, thickness, id_hl, err_type)
	# Show EST boxes.
	for box in box_list:
		if box.score >= 0.1:
			box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

 
	axes_limit = 35 + 3  # Slightly bigger to include boxes that extend beyond the range.
	ax.set_xlim(-axes_limit, axes_limit)
	ax.set_ylim(-2*axes_limit,5)
	plt.axis('off')

	plt.savefig(save_path)
	plt.close()
 

class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        # print(center.shape)
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)

    def render_cv2(self,
                   im: np.ndarray,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                   linewidth: int = 2) -> None:
        """
        Renders box using OpenCV2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: Linewidth for plot.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im,
                         (int(prev[0]), int(prev[1])),
                         (int(corner[0]), int(corner[1])),
                         color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(im,
                     (int(corners.T[i][0]), int(corners.T[i][1])),
                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                     colors[2][::-1], linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(im,
                 (int(center_bottom[0]), int(center_bottom[1])),
                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                 colors[0][::-1], linewidth)

    def copy(self) -> 'Box':
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)
