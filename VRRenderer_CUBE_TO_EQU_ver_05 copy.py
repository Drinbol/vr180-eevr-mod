import bpy
import os
import gpu
import bgl
import mathutils
import sys
import argparse
import numpy as np
from bpy.types import Operator, Panel
from math import sin, cos, pi
from datetime import datetime
from gpu_extras.batch import batch_for_shader


class VRRenderer:

    def __init__(self, is_test = 0, is_animation = False, mode = 'EQUI', FOV = 180):

        ################################################################################
        #レンダリング時正面片目解像度
        self.FrontResolution =2160

        #レンダリング時正面以外の片目解像度
        self.sideResolution = 1200

        #IPD設定(自分用メモ：0.2倍読み込み時=0.141くらい, 0.08倍読み込み時 = 0.0568くらいかな？)
        self.IPD= 0.0568

        #正面カメラ視野角(deg.) 境界線補正なし= 90, 補正あり = 108, もっと補正あり = 120
        self.front_angle = 108

        #有効カメラの設定
        #'left', 'right', 'bottom', 'top', 'front', 'back'
        self.render_bit = 0b111111
        ################################################################################
        if is_test != 0:
            self.render_bit = 0b0000010 #テストモード

        self.frong_angle_rag = self.front_angle*pi/180
        # Check if the file is saved or not, can cause errors when not saved
        if not bpy.data.is_saved:
            raise PermissionError("Save file before rendering")

        # Set internal variables for the class
        self.camera = bpy.context.scene.camera
        self.path = bpy.path.abspath("//")

        self.is_animation = is_animation
        self.FOV = FOV
        self.no_back_image = (self.FOV <= 270)
        self.no_side_images = (self.FOV <= 90) # TODO - Not implemented yet
        self.is_dome = (mode == 'DOME')
        self.createdFiles = set()

        # Set the image/folder name to the current time
        self.start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.folder_name = "Render Result {}/".format(self.start_time)

        # Get initial camera and output information
        self.camera_rotation = list(self.camera.rotation_euler)
        #self.IPD = self.camera.data.stereo.interocular_distance
        self.camera_type = self.camera.data.type
        self.stereo_type = self.camera.data.stereo.convergence_mode
        self.stereo_pivot = self.camera.data.stereo.pivot
        self.camera_angle = self.camera.data.angle

        # Create an empty to be used to control the camera
        if 'eeVR_CAMERA_EMPTY' in bpy.data.objects.keys():
            self.camera_empty = bpy.data.objects['eeVR_CAMERA_EMPTY']
        else:
            self.camera_empty = bpy.data.objects.new('eeVR_CAMERA_EMPTY', None)

        # Create a copy transforms constraint for the camera
        self.trans_constraint = self.camera.constraints.new('COPY_TRANSFORMS')

        #self.camera.rotation_euler

        # Set camera variables for proper result
        self.camera.data.type = 'PANO'
        self.camera.data.stereo.convergence_mode = 'PARALLEL'
        self.camera.data.stereo.pivot = 'CENTER'
        self.camera.data.sensor_fit = 'AUTO'


        self.tmp_image_size = [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y]
        #self.image_size = [self.combineResolution_x, self.combineResolution_y]
        #self.combineResolution = self.FrontResolution if self.FrontResolution > self.sideResolution else self.sideResolution

        #self.side_resolution = int(max(self.image_size)+4-max(self.image_size)%4)/2 if max(self.image_size)%4 > 0\
        #                    else int(max(self.image_size)/2)
        self.view_format = bpy.context.scene.render.image_settings.views_format
        bpy.context.scene.render.image_settings.views_format = 'STEREO_3D'
        self.stereo_mode = bpy.context.scene.render.image_settings.stereo_3d_format.display_mode
        bpy.context.scene.render.image_settings.stereo_3d_format.display_mode = 'TOPBOTTOM'

        self.direction_offsets = self.find_direction_offsets()
        #if self.no_back_image:
        #    fract = (self.FOV-90)/180
        #    self.camera_shift = {'top':[0.0, 0.5*(fract-1), self.side_resolution, fract*self.side_resolution],\
        #                        'bottom':[0.0, 0.5*(1-fract), self.side_resolution, fract*self.side_resolution],\
        #                        'left':[0.5*(1-fract), 0.0, fract*self.side_resolution, self.side_resolution],\
        #                        'right':[0.5*(fract-1), 0.0, fract*self.side_resolution, self.side_resolution],\
        #                        'front':[0.0, 0.0, self.side_resolution, self.side_resolution]}

        if self.no_back_image:
            fract = (self.FOV-90)/180
            self.camera_shift = {'top':[0.0, 0.5*(fract-1), self.sideResolution, fract*self.sideResolution,pi/2],\
                                'bottom':[0.0, 0.5*(1-fract), self.sideResolution, fract*self.sideResolution,pi/2],\
                                'left':[0.5*(1-fract), 0.0, fract*self.sideResolution, self.sideResolution,pi/2],\
                                'right':[0.5*(fract-1), 0.0, fract*self.sideResolution, self.sideResolution,pi/2],\
                                'front':[0.0, 0.0, self.FrontResolution, self.FrontResolution,self.frong_angle_rag]}


    def find_direction_offsets(self):

        # Calculate the pointing directions of the camera for each face of the cube
        # Using euler.rotate_axis() to handle, notice that rotation should be done on copies
        eul = self.camera.matrix_world.to_euler().copy()
        direction_offsets = {}
        #front
        direction_offsets['front'] = list(eul)
        #back
        eul.rotate_axis('Y', pi)
        direction_offsets['back'] = list(eul)
        #top
        eul = self.camera.matrix_world.to_euler().copy()
        eul.rotate_axis('X', pi/2)
        direction_offsets['top'] = list(eul)
        #bottom
        eul.rotate_axis('X', pi)
        direction_offsets['bottom'] = list(eul)
        #left
        eul = self.camera.matrix_world.to_euler().copy()
        eul.rotate_axis('Y', pi/2)
        direction_offsets['left'] = list(eul)
        #right
        eul.rotate_axis('Y', pi)
        direction_offsets['right'] = list(eul)
        return direction_offsets

    def set_camera_direction(self, direction):

        # Set the camera to the required postion
        self.camera_empty.rotation_euler = self.direction_offsets[direction]

        if self.no_back_image:
            self.camera.data.shift_x = self.camera_shift[direction][0]
            self.camera.data.shift_y = self.camera_shift[direction][1]
            bpy.context.scene.render.resolution_x = int(self.camera_shift[direction][2])
            bpy.context.scene.render.resolution_y = int(self.camera_shift[direction][3])
            self.camera.data.angle = self.camera_shift[direction][4]

    def clean_up(self):

        # Reset all the variables that were changed
        self.camera.constraints.remove(self.trans_constraint)
        self.camera.data.type = self.camera_type
        self.camera.data.stereo.convergence_mode = self.stereo_type
        self.camera.data.stereo.pivot = self.stereo_pivot
        self.camera.data.angle = self.camera_angle
        self.camera.rotation_euler = self.camera_rotation
        self.camera.data.shift_x = 0
        self.camera.data.shift_y = 0
        bpy.context.scene.render.resolution_x = self.tmp_image_size[0]
        bpy.context.scene.render.resolution_y = self.tmp_image_size[1]
        bpy.context.scene.render.image_settings.views_format = self.view_format
        bpy.context.scene.render.image_settings.stereo_3d_format.display_mode = self.stereo_mode


    def render_image(self, direction, extention):
        # Render the image and load it into the script
        tmp = bpy.context.scene.render.filepath

        image_name =  tmp +"/L/" +direction+"/" + \
            direction+"_frame{0:06d}{1}".format(bpy.context.scene.frame_current, extention)

        bpy.context.scene.render.filepath = image_name
        tmp_loc = list(self.camera_empty.location)
        camera_angle_y = self.direction_offsets['front'][1]
        camera_angle_z = self.direction_offsets['front'][2]
        self.camera_empty.location = [tmp_loc[0]-(0.5*self.IPD*cos(camera_angle_y)*cos(camera_angle_z)),\
                                    tmp_loc[1]-(0.5*self.IPD*cos(camera_angle_y)*sin(camera_angle_z)),\
                                    tmp_loc[2]+(0.5*self.IPD*sin(camera_angle_y))]
        bpy.ops.render.render(write_still=True)


        image_name =  tmp +"/R/" +direction+"/" + \
            direction+"_frame{0:06d}{1}".format(bpy.context.scene.frame_current, extention)

        bpy.context.scene.render.filepath = image_name
        self.camera_empty.location = [tmp_loc[0]+(0.5*self.IPD*cos(camera_angle_y)*cos(camera_angle_z)),\
                                    tmp_loc[1]+(0.5*self.IPD*cos(camera_angle_y)*sin(camera_angle_z)),\
                                    tmp_loc[2]-(0.5*self.IPD*sin(camera_angle_y))]

        bpy.ops.render.render(write_still=True)
        self.camera_empty.location = tmp_loc


        bpy.context.scene.render.filepath = tmp
        return


    def render_images(self):
        # Render the images for every direction
        directions = ['left', 'right', 'bottom', 'top', 'front', 'back']

        
        if bpy.context.scene.render.image_settings.file_format == 'PNG':
            extention = '.png'
        elif bpy.context.scene.render.image_settings.file_format == 'JPEG':
            extention = '.jpg'
        elif bpy.context.scene.render.image_settings.file_format == 'TIFF':
            extention = '.tif'

        #場所の確保がチェックされていたら仮ファイルを生成する
        savePath = bpy.context.scene.render.filepath +'/'
        tmpImage_name = savePath + '#tmp_frame{0:06d}{1}'.format(bpy.context.scene.frame_current, extention)
        if os.path.exists(tmpImage_name):
            return

        if bpy.context.scene.render.use_placeholder:
            try:
                with open(tmpImage_name, mode='x') as f:
                    f.write('')
            except:
                pass

        self.direction_offsets = self.find_direction_offsets()
        self.camera_empty.location = self.camera.matrix_world.to_translation()
        self.camera_empty.rotation_euler = self.camera.matrix_world.to_euler()
        self.trans_constraint.target = self.camera_empty

        for direction in directions:
            #上書きチェックが無い場合にファイルが既に存在していたらスキップする
            savePath = bpy.context.scene.render.filepath +'/R/' + direction + '/'
            image_name = direction + "_frame{0:06d}{1}".format(bpy.context.scene.frame_current, extention)
            if not bpy.context.scene.render.use_overwrite:
                if os.path.exists(savePath + image_name):
                    continue

            if direction == 'left' and self.render_bit & 0x20 == 0:
                continue
            elif direction == 'right'and self.render_bit & 0x10 == 0:
                continue
            elif direction == 'bottom'and self.render_bit & 0x8 == 0:
                continue
            elif direction == 'top'and self.render_bit & 0x4 == 0:
                continue
            elif direction == 'front'and self.render_bit & 0x2 == 0:
                continue
            elif direction == 'back' and self.render_bit & 0x1 == 0:
                continue

            if direction == 'back' and self.no_back_image:
                continue
            else:
                self.set_camera_direction(direction)
                self.render_image(direction, extention)

        self.set_camera_direction('front')
        self.trans_constraint.target = None

        #tmpファイルがある場合は削除
        if os.path.exists(tmpImage_name):
            os.remove(tmpImage_name)

        return


    def render_and_save(self):
        # Render the images and return their names
        self.render_images()

        if self.is_animation:
            bpy.context.scene.frame_set(bpy.context.scene.frame_current + bpy.context.scene.frame_step)

class VRRendererCancel(Operator):
    """Render out the animation"""

    bl_idname = 'wl.render_cancel'
    bl_label = "Cancel the render"

    def execute(self, context):
        context.scene.cancelVRRenderer = True
        return {'FINISHED'}


class RenderImage(Operator):
    """Render out the animation"""

    bl_idname = 'wl.render_image'
    bl_label = "Render a single frame"

    def execute(self, context):
        print("VRRenderer: execute")

        mode = bpy.context.scene.renderModeEnum
        FOV = bpy.context.scene.renderFOV
        renderer = VRRenderer(bpy.context.scene.render.use_multiview, False, mode, FOV)
        renderer.render_and_save()
        renderer.clean_up()

        return {'FINISHED'}


class RenderAnimation(Operator):
    """Render out the animation"""

    bl_idname = 'wl.render_animation'
    bl_label = "Render the animation"

    def __del__(self):
        print("VRRenderer: end")

    def modal(self, context, event):
        if event.type in {'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            wm = context.window_manager
            wm.event_timer_remove(self._timer)

            if context.scene.cancelVRRenderer:
                self.cancel(context)
                return {'CANCELLED'}

            if event.type in {'ESC'}:
                self.cancel(context)
                return {'CANCELLED'}

            if bpy.context.scene.frame_current <= self.frame_end:
                print("VRRenderer: Rendering frame {}".format(bpy.context.scene.frame_current))
                self._renderer.render_and_save()
                self._timer = wm.event_timer_add(0.1, window=context.window)
            else:
                self.clean(context)
                if gl_withArgs:
                    bpy.ops.wm.quit_blender()
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        print("VRRenderer: execute")
        context.scene.cancelVRRenderer = False

        mode = bpy.context.scene.renderModeEnum
        FOV = bpy.context.scene.renderFOV
        self._renderer = VRRenderer(bpy.context.scene.render.use_multiview, True, mode, FOV)

        self.frame_end = bpy.context.scene.frame_end
        frame_start = bpy.context.scene.frame_start
        bpy.context.scene.frame_set(frame_start)

        wm = context.window_manager
        self._timer = wm.event_timer_add(5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        print("VRRenderer: cancel")
        self.clean(context)

    def clean(self, context):
        self._renderer.clean_up()
        context.scene.cancelVRRenderer = True



class RenderToolsPanel(Panel):
    """Tools panel for VR rendering"""

    bl_idname = "RENDER_TOOLS_PT_render_tools_panel2"
    bl_label = "Render Tools"
    bl_category = "VRRenderer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):

        # Draw the buttons for each of the rendering operators
        layout = self.layout
        col = layout.column()
        col.prop(context.scene, 'renderModeEnum')
        col.prop(context.scene, 'renderFOV')
        col.operator("wl.render_image", text="Render Image")
        col.operator("wl.render_animation", text="Render Animation")
        if not context.scene.cancelVRRenderer:
            col.operator("wl.render_cancel", text="Cancel")
            col.label(text="Rendering frame {}".format(bpy.context.scene.frame_current))


# Register all classes
def register():
    bpy.types.Scene.renderModeEnum = bpy.props.EnumProperty(
        items = [('EQUI', 'Equirectangular', 'Renders in equirectangular projection'),
                ('DOME', 'Full Dome', 'Renders in full dome projection')],
        default='EQUI',
        name="Mode")
    bpy.types.Scene.renderFOV = bpy.props.FloatProperty(180.0,default=180.0, name="FOV", min=180, max=360,
                                description="Field of view in degrees")
    bpy.types.Scene.cancelVRRenderer = bpy.props.BoolProperty(name="Cancel", default=True)
    bpy.utils.register_class(RenderImage)
    bpy.utils.register_class(RenderAnimation)
    bpy.utils.register_class(RenderToolsPanel)
    bpy.utils.register_class(VRRendererCancel)




# Unregister all classes
def unregister():
    del bpy.types.Scene.renderModeEnum
    del bpy.types.Scene.renderFOV
    bpy.utils.unregister_class(RenderImage)
    bpy.utils.unregister_class(RenderAnimation)
    bpy.utils.unregister_class(RenderToolsPanel)
    bpy.utils.unregister_class(VRRendererCancel)


# If the script is not an addon when it is run, register the classes
if __name__=="__main__":
    register()

    parser = argparse.ArgumentParser()
    parser.add_argument('--FOV', type=int,choices=[180, 360])
    parser.add_argument('--startFrame', type=int)
    parser.add_argument('--endFrame', type=int)
    parser.add_argument('--step', type=int)
    parser.add_argument('--test', type=int)

    gl_withArgs = True
    try:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    except Exception as e:
        gl_withArgs = False

    #コマンドラインによる引数あり実行
    if gl_withArgs:
        step = args.step
        if step < 0:
            step = 1
        bpy.context.scene.frame_step = step
        bpy.context.scene.frame_end = args.endFrame
        bpy.context.scene.frame_start = args.startFrame
        bpy.context.scene.frame_set(args.startFrame)

        if bpy.app.background:
            mode = 'EQUI'
            FOV = args.FOV
            test = args.test
            _renderer = VRRenderer(test, True, mode, FOV)

            while bpy.context.scene.frame_current <= bpy.context.scene.frame_end:
                print("VRRenderer: Rendering frame {}".format(bpy.context.scene.frame_current))
                _renderer.render_and_save()
            _renderer.clean_up()

        else:
            bpy.ops.wl.render_animation()





