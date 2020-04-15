import bpy
import time
import sys
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import numpy as np


def addNode(tree, typ):
    node = tree.nodes.new(typ)
    return node


def linkNodes(tree, node1, node2, out=0, inp=0):
    tree = bpy.context.scene.node_tree
    tree.links.new(node1.outputs[out], node2.inputs[inp])


def scale(value, min, max):
    return min + (max - min) * value
    

class Blender_Setup():
    
    def __init__(self, pose, width, height):
        self.context = bpy.context
        self.scene = bpy.data.scenes['Scene']
        self.pose = pose
        self.width = width
        self.height = height
        self.scale_factor = 3
        self.world_bone_locations = []
        # initial positions in model by clicking on bone in x-ray
        self.initial_bone_positions = {
            "arm elbow_R": Vector((-.354, .0009, .132)),
            "arm elbow_L": Vector((.354, .0009, .132)),
            "heel_R": Vector((-.113, -.094, -1.14)),
            "heel_L": Vector((.113, -.094, -1.14)),
            "hip": Vector((0, .004, .027))}
        for key in self.initial_bone_positions:
            self.initial_bone_positions[
                key] = self.scale_factor*self.initial_bone_positions[key]

    def add_plane(self, name, location, rotation_euler, scale):
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.object
        obj.location = location
        obj.rotation_euler = rotation_euler
        obj.scale = scale
        obj.name = name

    def render(self, filepath):
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)

    def renderer(self, render_type):
        self.scene.render.resolution_x = self.width
        self.scene.render.resolution_y = self.height
        self.scene.render.resolution_percentage = 100
        if render_type == 'wire':
            self.setup_for_wireframe()
        elif render_type == 'depth':
            self.setup_for_depth()
        bpy.data.objects["Camera"].location = Vector([0,-8.5, 5])
        bpy.data.objects["rig"].location = Vector([0, 0, 0])
        bpy.data.objects["rig"].scale = Vector(
            [self.scale_factor, self.scale_factor, self.scale_factor])
        self.set_object_rotation_euler(
            "Camera", Vector([np.pi / 3., 0, 0]))
        self.add_plane("background", Vector([0, 4, 0]),
                       Vector([np.pi / 3., 0, 0]), Vector([20, 20, 20]))
        self.add_plane("nearplane",
                       Vector([-2, -4, 0]),
                       Vector([np.pi / 3., 0, 0]), Vector([0.1, 0.1, 0.1]))
        self.set_body_pose()
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
        tree = bpy.context.scene.node_tree

    def setup_for_wireframe(self):
        # using blender_eevee prevents random refraction off of the wire
        self.scene.render.engine = "BLENDER_EEVEE"
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                obj.modifiers.new("Wireframe", type="WIREFRAME")
                obj.modifiers["Wireframe"].thickness = .05
                obj.modifiers["Wireframe"].use_even_offset = False
                obj.modifiers["Wireframe"].use_replace = True
        
    def get_bone_location(self, object_name, bone_name):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        world_bone_location = Vector(
            self.initial_bone_positions[bone_name]) + bone.location
        self.world_bone_locations.append(world_bone_location)
        print(bone_name)
        print(world_bone_location)
        final_2d_coords = self.world_to_camera_view(bpy.data.objects["Camera"],
                                                    world_bone_location)
        return [final_2d_coords[0]*self.scene.render.resolution_x,
                final_2d_coords[1]*self.scene.render.resolution_y,
                final_2d_coords[2]]

    def get_bone_rotation_euler(self, object_name, bone_name):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        return tuple(bone.rotation_euler)

#    def recover_pose_variables(self, 

    def world_to_camera_view(self, cam, coords):
        co_local = cam.matrix_world.normalized().inverted() @ coords
        z = -co_local.z
        camera = cam.data
        frame = [-v for v in camera.view_frame()[:3]]
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
        cam = bpy.data.objects["Camera"]
        x, y, z_depth = depth_vec
        x /= self.scene.render.resolution_x
        y /= self.scene.render.resolution_y
        view_frame = [-v for v in cam.data.view_frame()[:3]]
        frame = [(v / (v.z / z_depth)) for v in view_frame]
        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y
        co_local = Vector()
        co_local.x = ((max_x - min_x) * x) + min_x
        co_local.y = ((max_y - min_y) * y) + min_y
        co_local.z = -z_depth
        xyz_location = cam.matrix_world.normalized() @ co_local
        return xyz_location

    def set_object_rotation_euler(self, name, rotation_euler):
        obj = bpy.data.objects[name]
        obj.rotation_euler = rotation_euler

    def set_bone_location(self, object_name, bone_name, location):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        bone.location = location

    def set_bone_rotation_euler(self, object_name, bone_name, rotation_euler):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = rotation_euler

    def set_body_pose(self):
        self.set_object_rotation_euler("rig", self.pose.rotation)
        self.set_bone_location("rig", "arm elbow_R", self.pose.elbow_r_loc)
        self.set_bone_location("rig", "arm elbow_L", self.pose.elbow_l_loc)
        self.set_bone_rotation_euler(
            "rig", "arm elbow_R", self.pose.elbow_r_rot)
        self.set_bone_rotation_euler(
            "rig", "arm elbow_L", self.pose.elbow_l_rot)
        self.set_bone_location("rig", "hip", self.pose.hip_loc)
        self.set_bone_location("rig", "heel_R", self.pose.heel_r_loc)
        self.set_bone_location("rig", "heel_L", self.pose.heel_l_loc)

# here just pass pose statistics and then set_body_pose, then render.

# Put all pose variables into the generative model. without that,
# it seems like the model has no structure. 

class BodyPose():
    def __init__(self, pose_dict):
        self.rotation = Vector([0, 0, scale(pose_dict['rot_z'],
                                            -np.pi / 4, np.pi / 4)])
       # self.rotation = Vector([0, 0, 0])
        self.elbow_r_loc = Vector([scale(pose_dict['elbow_r_loc_x'], -1, 1),
                                   scale(pose_dict['elbow_r_loc_y'], -1, 1),
                                   scale(pose_dict['elbow_r_loc_z'], -1, 1)])
        self.elbow_r_rot = Vector([0, 0, scale(pose_dict['elbow_r_rot'],
                                               0, 2*np.pi)])
        self.elbow_l_loc = Vector([scale(pose_dict['elbow_l_loc_x'], -1, 1),
                                   scale(pose_dict['elbow_l_loc_y'], -1, 1),
                                   scale(pose_dict['elbow_l_loc_z'], -1, 1)])
        self.elbow_l_rot = Vector([0, 0, scale(pose_dict['elbow_l_rot'],
                                               0, 2*np.pi)])
        self.hip_loc = Vector([0, 0, scale(pose_dict['hip_loc_z'], -.35, 0)])
        self.heel_l_loc = Vector([scale(pose_dict['heel_l_loc_x'], -.5, .5),
                                  scale(pose_dict['heel_l_loc_y'], -1, .5),
                                  scale(pose_dict['heel_l_loc_z'], -.2, .2)])
        self.heel_r_loc = Vector([scale(pose_dict['heel_r_loc_x'], -.5, .5),
                                  scale(pose_dict['heel_r_loc_y'], -1, .5),
                                  scale(pose_dict['heel_r_loc_z'], -.2, .2)])

# eventually replace the random calls with calls from the generative model
# this is just infinitely easier to have in the python file      
# don't have to worry about this on the other end...
# all you have to do is unscale findings from the neural net.
pose_dict_values = [float(pd) for pd in sys.argv[sys.argv.index("--") + 1:-1]]


# note as long as heel is 

pose_dict_keys = ['rot_z',
                  'elbow_r_loc_x',
                  'elbow_r_loc_y',
                  'elbow_r_loc_z',
                  'elbow_l_loc_x',
                  'elbow_l_loc_y',
                  'elbow_l_loc_z',
                  'elbow_r_rot',
                  'elbow_l_rot',
                  'hip_loc_z',
                  'heel_r_loc_x',
                  'heel_r_loc_y',
                  'heel_r_loc_z',
                  'heel_l_loc_x',
                  'heel_l_loc_y',
                  'heel_l_loc_z']

joints = ["arm elbow_R",
          "arm elbow_L",
          "hip",
          "heel_R",
          "heel_L"]

pose_dict = dict(zip(pose_dict_keys, pose_dict_values))
body_pose = BodyPose(pose_dict)
if sys.argv[-1] == "depth":
  #  width, height = (256, 256)
    width, height = (256, 256)
elif sys.argv[-1] == "wire":
    width, height = (512, 512)
blender_creator = Blender_Setup(body_pose, width, height)
blender_creator.renderer(sys.argv[-1])

joint_positions_2d = [blender_creator.get_bone_location("rig", j) for j in joints]

recovered_bone_locations = [blender_creator.world_from_depth_coords(j) for j in joint_positions_2d]

pose_recovery = [rbl - blender_creator.initial_bone_positions[j] for j, rbl in zip(joints, recovered_bone_locations)]

                               # GOOD THIS WORKS! HAVE TO RECOVER POSE DICT NOW FROM 

print("Recovered")
print(pose_recovery)
print("Original")
print(body_pose.elbow_r_loc)

# OK so now everything works. You can get 2d coordinates from pose input and recover pose input
# from 2d depth coords. All you have to do is unscale to get to the original Gen choices.
# think about how to do this elegantly. 






# joint_file = open(sys.argv[-1]+'.txt', 'w')
# joint_file.writelines([str(j)[1:-1]+"\n" for j in joint_positions_2d])
# true_positions = [blender_creator.world_from_depth_coords(jointxyd) - Vector(blender_creator.initial_bone_positions[joint]) for joint, jointxyd in zip(joints, joint_positions_2d)]
# print("TRUE POSITIONS")
# print(true_positions)
# joint_file.close()
# save 


                                 
# all you have to expose is the class to the rpyc client.
# should be fine -- may have to expose the
# depth_renderer and wireframe_renderer, keeping everything else hidden

