import bpy
import time
import sys
from mathutils import Vector
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
    
    def __init__(self, pose):
        self.context = bpy.context
        self.scene = bpy.data.scenes['Scene']
        self.pose = pose

    def renderer(self, width, height, render_type):
        self.set_resolution(width, height)
        if render_type == 'wire':
            self.setup_for_wireframe()
        elif render_type == 'depth':
            self.setup_for_depth()

        self.set_object_location("Camera", Vector([0, -8.5, 5]))
        self.set_object_rotation_euler(
            "Camera", Vector([np.pi / 3., 0, 0]))
        self.add_plane("background", Vector([0, 4, 0]),
                       Vector([np.pi / 3., 0, 0]), Vector([20, 20, 20]))
        self.set_object_location("rig", Vector([0, 0, 0]))
     #   self.set_object_rotation_euler("rig", Vector([0, 0, 0]))
        self.set_object_scale("rig", Vector([3, 3, 3]))
        self.add_plane("nearplane",
                       Vector([-2, -4, 0]),
                       Vector([np.pi / 3., 0, 0]), Vector([0.1, 0.1, 0.1]))
        self.set_body_pose()
        try:
            os.remove(render_type+".png")
        except NameError:
            pass
        self.render(render_type+".png")
        

    #    self.setup_for_depth()
    #    self.set_resolution(width, height)

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
        addNode(tree, 'CompositorNodeNormalize')
        render = tree.nodes['Render Layers']
        composite = tree.nodes['Composite']
        norm = tree.nodes['Normalize']
        linkNodes(tree, render, norm, out=2)
        linkNodes(tree, norm, composite)
    #        bpy.data.worlds['World'].horizon_color = (0, 0, 0)
        bpy.data.worlds['World'].color = (0, 0, 0)
        self.scene.render.resolution_percentage = 100

    def setup_for_wireframe(self):
        # using blender_eevee prevents random refraction off of the wire
        self.scene.render.engine = "BLENDER_EEVEE"
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                obj.modifiers.new("Wireframe", type="WIREFRAME")
                obj.modifiers["Wireframe"].thickness = .05
                obj.modifiers["Wireframe"].use_even_offset = False
                obj.modifiers["Wireframe"].use_replace = True

    def set_resolution(self, x, y):
        self.scene.render.resolution_x = x
        self.scene.render.resolution_y = y
        self.scene.render.resolution_percentage = 100

    def add_plane(self, name, location, rotation_euler, scale):
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.object
        obj.location = location
        obj.rotation_euler = rotation_euler
        obj.scale = scale
        obj.name = name
        
    def set_object_location(self, name, location):
        obj = bpy.data.objects[name]
        obj.location = location

    def set_object_rotation_euler(self, name, rotation_euler):
        obj = bpy.data.objects[name]
        obj.rotation_euler = rotation_euler

    def set_object_scale(self, name, scale):
        obj = bpy.data.objects[name]
        obj.scale = scale

    def get_object_location(self, name):
        return tuple(bpy.data.objects[name].location)

    def get_object_rotation_euler(self, name):
        return tuple(bpy.data.objects[name].rotation_euler)

    def get_object_scale(self, name):
        return tuple(bpy.data.objects[name].scale)

    def render(self, filepath):
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)

    def set_bone_location(self, object_name, bone_name, location):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        bone.location = location

    def set_bone_rotation_euler(self, object_name, bone_name, rotation_euler):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = rotation_euler

    def get_bone_location(self, object_name, bone_name):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        return tuple(bone.location)

    def get_bone_rotation_euler(self, object_name, bone_name):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        return tuple(bone.rotation_euler)

  #   def get_body_pose(client::BlenderClient)
#     BodyPose(
#         get_object_rotation_euler(client, RIG),
#         get_bone_location(client, RIG, ARM_ELBOW_R),
#         get_bone_location(client, RIG, ARM_ELBOW_L),
#         get_bone_rotation_euler(client, RIG, ARM_ELBOW_R),
#         get_bone_rotation_euler(client, RIG, ARM_ELBOW_L),
#         get_bone_location(client, RIG, HIP),
#         get_bone_location(client, RIG, HEEL_R),
#         get_bone_location(client, RIG, HEEL_L))
# end
    
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
      

class BodyPose_self_generating():
    def __init__(self):
        rotation_z = np.random.uniform(0, 1)
        self.rotation = Vector([0, 0, scale(rotation_z, -np.pi / 4, np.pi / 4)])
        self.elbow_r_loc = Vector([scale(np.random.uniform(0, 1), -1, 1),
                                   scale(np.random.uniform(0, 1), -1, 1),
                                   scale(np.random.uniform(0, 1), -1, 1)])
        self.elbow_r_rot = Vector([0, 0, scale(np.random.uniform(0, 1), 0, 2*np.pi)])
        self.elbow_l_loc = Vector([scale(np.random.uniform(0, 1), -1, 1),
                                   scale(np.random.uniform(0, 1), -1, 1),
                                   scale(np.random.uniform(0, 1), -1, 1)])
        self.elbow_l_rot = Vector([0, 0, scale(np.random.uniform(0, 1), 0, 2*np.pi)])
        self.hip_loc = Vector([0, 0, scale(np.random.uniform(0, 1), -.35, 0)])
        self.heel_l_loc = Vector([scale(np.random.uniform(0, 1), -.5, .5),
                                  scale(np.random.uniform(0, 1), -1, .5),
                                  scale(np.random.uniform(0, 1), -.2, .2)])
        self.heel_r_loc = Vector([scale(np.random.uniform(0, 1), -.5, .5),
                                  scale(np.random.uniform(0, 1), -1, .5),
                                  scale(np.random.uniform(0, 1), -.2, .2)])



class BodyPose():
    def __init__(self, pose_dict):
        self.rotation = Vector([0, 0, scale(pose_dict['rot_z'],
                                            -np.pi / 4, np.pi / 4)])
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




pose_dict = dict(zip(pose_dict_keys, pose_dict_values))
body_pose = BodyPose(pose_dict)
blender_creator = Blender_Setup(body_pose)
if sys.argv[-1] == "depth":
    width, height = (128, 128)
elif sys.argv[-1] == "wire":
    width, height = (512, 512)
blender_creator.renderer(width, height, sys.argv[-1])


                                 
# all you have to expose is the class to the rpyc client.
# should be fine -- may have to expose the
# depth_renderer and wireframe_renderer, keeping everything else hidden

