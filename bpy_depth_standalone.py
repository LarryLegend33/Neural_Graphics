import bpy
import time
import sys
from mathutils import Vector, Matrix, Euler
from bpy_extras.object_utils import world_to_camera_view
import numpy as np


class Blender_Setup():
    
    def __init__(self, pose, width, height, joints):
        self.context = bpy.context
        self.scene = bpy.data.scenes['Scene']
        self.pose = pose
        self.width = width
        self.height = height
        self.scale_factor = 3
        self.joints = joints
        self.camera = bpy.data.objects["Camera"]
        self.rig = bpy.data.objects["rig"]
        self.bone_locations_after_pose = {}
        self.groundtruth_xy = {}
        self.bone = (lambda x: self.rig.pose.bones[x])
        self.bone_world_matrix = (lambda x: self.rig.matrix_world @ self.bone(
            x).matrix)
        self.bone_location = (lambda x: self.bone_world_matrix(
            x) @ self.bone(x).location)

    def add_plane(self, name, location, rotation_euler, scale):
        bpy.ops.mesh.primitive_plane_add()
        obj = self.context.object
        obj.location = location
        obj.rotation_euler = rotation_euler
        obj.scale = scale
        obj.name = name

    # there is a factor of 2 off here somewhere. if you add a bump, it doubles it.
    # setting directly doesn't work. have to bump. 
    def render(self, filepath):
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        
    def renderer(self, render_type):
        self.scene.render.resolution_x = self.width
        self.scene.render.resolution_y = self.height
        self.bone("arm elbow_R").rotation_mode = 'XYZ'
        self.bone("arm elbow_L").rotation_mode = 'XYZ'        
        self.scene.render.resolution_percentage = 100
        if render_type == 'wire':
            self.setup_for_wireframe()
        elif render_type == 'depth':
            self.setup_for_depth()
        self.camera.location = Vector([0, -8.5, 5])
        self.rig.location = Vector([0, 0, 0])
        self.rig.scale = Vector(
            [self.scale_factor, self.scale_factor, self.scale_factor])
        self.camera.rotation_euler = Euler([np.pi / 3., 0, 0], 'XYZ')
        self.add_plane("background", Vector([0, 4, 0]),
                       Euler([np.pi / 3., 0, 0], 'XYZ'), Vector([20, 20, 20]))
        self.add_plane("nearplane",
                       Vector([-2, -4, 0]),
                       Euler([np.pi / 3., 0, 0], 'XYZ'), Vector([0.1, 0.1, 0.1]))
        self.context.view_layer.update()
        self.set_body_pose()
        self.get_bone_locations()
        try:
            os.remove(render_type+".png")
        except NameError:
            pass
        self.render(render_type+".png")

    def setup_for_depth(self):
        self.scene.render.engine = "CYCLES"
        self.scene.cycles.samples = 1
        self.scene.render.tile_x = 25
        self.scene.render.tile_y = 25
        self.scene.cycles.max_bounces = 0
        self.scene.cycles.caustics_reflective = False
        self.scene.cycles.caustics_refractive = False
        # RenderSettings eliminated these properties in 2.8.
        # Had to switch to view_layer
     #   self.scene.render.layers[0].use_pass_normal = False
          #   self.scene.render.layers[0].use_pass_combined = False
        self.context.view_layer.use_pass_normal = False
        self.context.view_layer.use_pass_combined = False
        self.scene.render.use_compositing = True
        self.scene.use_nodes = True
        tree = self.context.scene.node_tree

    def setup_for_wireframe(self):
        # using blender_eevee prevents random refraction off of the wire
        self.scene.render.engine = "BLENDER_EEVEE"
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                obj.modifiers.new("Wireframe", type="WIREFRAME")
                obj.modifiers["Wireframe"].thickness = .05
                obj.modifiers["Wireframe"].use_even_offset = False
                obj.modifiers["Wireframe"].use_replace = True

    # delta xyz isnt in terms of joints' basis. its in real world XYZ space. 
    def update_bone_location(self, bone_id, delta_xyz):
        new_xyz = self.bone_location(bone_id) + delta_xyz
        self.bone(bone_id).location = self.bone_world_matrix(
            bone_id).inverted() @ new_xyz
        bpy.context.view_layer.update()

    def get_bone_locations(self):
        resolution_matrix = Matrix(((self.scene.render.resolution_x, 0, 0),
                                    (0, self.scene.render.resolution_y, 0),
                                    (0, 0, 1)))
        self.bone_locations_after_pose = {
            j: self.bone_location(j) for j in joints}
        self.groundtruth_xy = {
            j: resolution_matrix @ self.world_to_camera_view(
                self.bone_locations_after_pose[j]) for j in self.joints}
#        print(self.bone_locations_after_pose)
#        print(self.groundtruth_xy)

    def world_to_camera_view(self, coords):
        co_local = self.camera.matrix_world.normalized().inverted() @ coords
        z = -co_local.z
        frame = [-v for v in self.camera.data.view_frame(scene=self.scene)[:3]]
        if z == 0.0:
            return Vector((0.5, 0.5, 0.0))
        else:
            frame = [(v / (v.z / z)) for v in frame]
        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y
        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)
        return Vector((x, y, z))
      
    def world_from_depth_coords(self, depth_vec):
        x, y, z_depth = depth_vec
        x /= self.scene.render.resolution_x
        y /= self.scene.render.resolution_y
        view_frame = [-v for v in self.camera.data.view_frame()[:3]]
        frame = [(v / (v.z / z_depth)) for v in view_frame]
        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y
        co_local = Vector()
        co_local.x = ((max_x - min_x) * x) + min_x
        co_local.y = ((max_y - min_y) * y) + min_y
        co_local.z = -z_depth
        xyz_location = self.camera.matrix_world.normalized() @ co_local
        return xyz_location


    # WORKS!!! YOU GET THE EXACT RIGHT BUMP FROM THE POSE. 
    def set_body_pose(self):
        self.rig.rotation_euler = Euler([0, 0, self.pose['rot_z']], 'XYZ')
        print(self.bone_location("arm elbow_R"))
        print(Vector([self.pose['elbow_r_x'] / 2,
                      self.pose['elbow_r_y'] / 2,
                      self.pose['elbow_r_z'] / 2]))
        self.update_bone_location("arm elbow_R",
                                  Vector([self.pose['elbow_r_x'] / 2,
                                          self.pose['elbow_r_y'] / 2,
                                          self.pose['elbow_r_z'] / 2]))
        print(self.bone_location("arm elbow_R"))
        # self.bone("arm elbow_R").location = Vector([self.pose['elbow_r_x'],
        #                                             self.pose['elbow_r_y'],
        #                                             self.pose['elbow_r_z']])
        # self.bone("arm elbow_L").location = Vector([self.pose['elbow_l_x'],
        #                                             self.pose['elbow_l_y'],
        #                                             self.pose['elbow_l_z']])
        # self.bone("hip").location = Vector([0, 0, self.pose['hip_z']])
        # self.bone("heel_R").location = Vector([self.pose['heel_r_x'],
        #                                        self.pose['heel_r_y'],
        #                                        self.pose['heel_r_z']])
        # self.bone("heel_L").location = Vector([self.pose['heel_l_x'],
        #                                        self.pose['heel_l_y'],
        #                                        self.pose['heel_l_z']])
        # self.bone("arm elbow_R").rotation_euler = Euler(
        #     [0, 0, self.pose['elbow_r_rot']], 'XYZ')
        # self.bone("arm elbow_L").rotation_euler = Euler(
        #     [0, 0, self.pose['elbow_l_rot']], 'XYZ')
       
        self.context.view_layer.update()

# here just pass pose statistics and then set_body_pose, then render.
# this class organizes the stochastic choices into vectors

pose_dict_values = [float(pd) for pd in sys.argv[sys.argv.index("--") + 1:-1]]
pose_dict_keys = ['rot_z',
                  'elbow_r_x',
                  'elbow_r_y',
                  'elbow_r_z',
                  'elbow_l_x',
                  'elbow_l_y',
                  'elbow_l_z',
                  'elbow_r_rot',
                  'elbow_l_rot',
                  'hip_z',
                  'heel_r_x',
                  'heel_r_y',
                  'heel_r_z',
                  'heel_l_x',
                  'heel_l_y',
                  'heel_l_z']

#arm2 is relatively accurate. arm1 is perfect (shoulder)
#heel r and l are both perfect
# hip is perfect
# hands are constant no matter what the pose. (??)

joints = ["arm elbow_R",
          "arm elbow_L",
          "hip",
          "heel_R",
          "heel_L"]

pose_dict = dict(zip(pose_dict_keys, pose_dict_values))
if sys.argv[-1] == "depth":
  #  width, height = (256, 256)
    width, height = (256, 256)
elif sys.argv[-1] == "wire":
    width, height = (512, 512)
blender_creator = Blender_Setup(pose_dict, width, height, joints)
blender_creator.renderer(sys.argv[-1])
joint_file = open(sys.argv[-1]+'.txt', 'w')
joint_file.writelines([str(
    tuple(blender_creator.groundtruth_xy[j]))[1:-1]+"\n" for j in joints])
joint_file.close()


