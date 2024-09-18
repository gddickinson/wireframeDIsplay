import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QPushButton,
                             QSlider, QLabel, QMessageBox, QGroupBox, QCheckBox, QColorDialog, QAction,
                             QFileDialog, QSpinBox, QListWidget, QSplitter, QDialog)
from PyQt5.QtCore import Qt, QTime, QTimer
from PyQt5.QtGui import QColor
import pyqtgraph.opengl as gl
import numpy as np
from wireframe import Wireframe
from shape_loader_dialog import ShapeLoaderDialog

class SceneObject:
    def __init__(self, wireframe, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        self.wireframe = wireframe
        self.position = np.array(position, dtype=float)
        self.rotation = np.array(rotation, dtype=float)
        self.scale = np.array(scale, dtype=float)
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.mesh_item = None
        self.edge_item = None
        self.vertex_item = None

    def update_mesh(self, display_options):
        if self.mesh_item is None:
            self.mesh_item = gl.GLMeshItem(vertexes=self.wireframe.vertices, faces=self.wireframe.faces,
                                           smooth=True, drawEdges=True, edgeColor=(0, 1, 0, 1))
        self.apply_transformation()

        # Update display options
        self.mesh_item.setVisible(display_options.show_faces.isChecked())
        self.mesh_item.setColor(display_options.face_color_btn.color.getRgbF())

        if display_options.show_edges.isChecked():
            if self.edge_item is None:
                self.edge_item = gl.GLLinePlotItem()
            self.edge_item.setData(pos=self.wireframe.vertices, color=display_options.edge_color_btn.color.getRgbF(),
                                   width=display_options.edge_width.value(), antialias=True, mode='lines')
        else:
            self.edge_item = None

        if display_options.show_vertices.isChecked():
            if self.vertex_item is None:
                self.vertex_item = gl.GLScatterPlotItem()
            self.vertex_item.setData(pos=self.wireframe.vertices, color=display_options.vertex_color_btn.color.getRgbF(),
                                     size=display_options.vertex_size.value())
        else:
            self.vertex_item = None

    def apply_transformation(self):
        # Apply scale
        scaled_vertices = self.wireframe.vertices * self.scale

        # Apply rotation
        rotation_x = np.array([[1, 0, 0],
                               [0, np.cos(self.rotation[0]), -np.sin(self.rotation[0])],
                               [0, np.sin(self.rotation[0]), np.cos(self.rotation[0])]])
        rotation_y = np.array([[np.cos(self.rotation[1]), 0, np.sin(self.rotation[1])],
                               [0, 1, 0],
                               [-np.sin(self.rotation[1]), 0, np.cos(self.rotation[1])]])
        rotation_z = np.array([[np.cos(self.rotation[2]), -np.sin(self.rotation[2]), 0],
                               [np.sin(self.rotation[2]), np.cos(self.rotation[2]), 0],
                               [0, 0, 1]])
        rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
        rotated_vertices = np.dot(scaled_vertices, rotation_matrix.T)

        # Apply translation
        translated_vertices = rotated_vertices + self.position

        self.mesh_item.setMeshData(vertexes=translated_vertices, faces=self.wireframe.faces)

class Scene:
    def __init__(self):
        self.objects = []

    def add_object(self, scene_object):
        self.objects.append(scene_object)

    def remove_object(self, scene_object):
        self.objects.remove(scene_object)

    def update_physics(self, dt):
        for obj in self.objects:
            # Update position
            obj.position += obj.velocity * dt

            # Update rotation
            obj.rotation += obj.angular_velocity * dt

            # Apply simple gravity
            obj.velocity[1] -= 9.8 * dt  # Assuming Y is up

            # Simple ground collision
            if obj.position[1] < 0:
                obj.position[1] = 0
                obj.velocity[1] = -obj.velocity[1] * 0.8  # Bounce with damping

    def apply_transformations(self):
        for obj in self.objects:
            obj.apply_transformation()


class ColorButton(QPushButton):
    def __init__(self, color):
        super().__init__()
        self.setColor(color)

    def setColor(self, color):
        self.color = color
        self.setStyleSheet(f"background-color: {color.name()}; color: {'black' if color.lightness() > 127 else 'white'}")
        self.setText(color.name())

class DisplayOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Display Options")
        self.setGeometry(100, 100, 300, 300)

        layout = QVBoxLayout()

        self.show_faces = QCheckBox("Show Faces")
        self.show_edges = QCheckBox("Show Edges")
        self.show_vertices = QCheckBox("Show Vertices")

        self.face_color_btn = ColorButton(QColor(200, 200, 200))
        self.edge_color_btn = ColorButton(QColor(0, 0, 0))
        self.vertex_color_btn = ColorButton(QColor(255, 0, 0))

        self.edge_width = QSpinBox()
        self.edge_width.setRange(1, 10)
        self.edge_width.setValue(1)
        self.edge_width.setPrefix("Edge Width: ")

        self.vertex_size = QSpinBox()
        self.vertex_size.setRange(1, 20)
        self.vertex_size.setValue(5)
        self.vertex_size.setPrefix("Vertex Size: ")

        layout.addWidget(self.show_faces)
        layout.addWidget(self.show_edges)
        layout.addWidget(self.show_vertices)
        layout.addWidget(self.face_color_btn)
        layout.addWidget(self.edge_color_btn)
        layout.addWidget(self.vertex_color_btn)
        layout.addWidget(self.edge_width)
        layout.addWidget(self.vertex_size)

        self.setLayout(layout)

class WireframeViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scene = Scene()
        self.initUI()
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_scene)
        self.animation_timer.start(16)  # ~60 FPS
        self.last_time = QTime.currentTime()
        self.is_playing = False  # New attribute to track if animation is playing


    def initUI(self):
        self.setWindowTitle('Wireframe Viewer')
        self.setGeometry(100, 100, 1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel for object list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.object_list = QListWidget()
        left_layout.addWidget(QLabel("Scene Objects:"))
        left_layout.addWidget(self.object_list)

        load_button = QPushButton('Load Shape')
        load_button.clicked.connect(self.show_shape_loader)
        left_layout.addWidget(load_button)

        remove_button = QPushButton('Remove Selected')
        remove_button.clicked.connect(self.remove_selected_object)
        left_layout.addWidget(remove_button)

        # Animation controls
        anim_layout = QHBoxLayout()
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.toggle_animation)
        anim_layout.addWidget(self.play_button)
        left_layout.addLayout(anim_layout)

        splitter.addWidget(left_panel)

        # Right panel for 3D view and controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 3D visualization widget
        self.view = gl.GLViewWidget()
        right_layout.addWidget(self.view, stretch=4)

        # Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        right_layout.addWidget(controls_widget, stretch=1)

        # Transformation controls
        self.create_transformation_controls(controls_layout)

        splitter.addWidget(right_panel)

        # Set the initial sizes of the splitter
        splitter.setSizes([200, 800])

        # Menu
        self.create_menus()

        # Shape loader
        self.shape_loader = ShapeLoaderDialog(self)
        self.shape_loader.shape_loaded.connect(self.on_shape_loaded)

        # Display options
        self.display_options = DisplayOptionsDialog(self)
        self.setup_display_options()

    def create_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        view_menu = menubar.addMenu('View')
        shape_menu = menubar.addMenu('Shapes')

        open_action = QAction('Open', self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction('Save', self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        display_options_action = QAction('Display Options', self)
        display_options_action.triggered.connect(self.show_display_options)
        view_menu.addAction(display_options_action)

        load_shape_action = QAction('Load Shape', self)
        load_shape_action.triggered.connect(self.show_shape_loader)
        shape_menu.addAction(load_shape_action)

    def setup_display_options(self):
        self.display_options.show_faces.setChecked(True)
        self.display_options.show_edges.setChecked(True)
        self.display_options.show_vertices.setChecked(False)
        self.display_options.face_color_btn.clicked.connect(lambda: self.change_color('face'))
        self.display_options.edge_color_btn.clicked.connect(lambda: self.change_color('edge'))
        self.display_options.vertex_color_btn.clicked.connect(lambda: self.change_color('vertex'))
        self.display_options.show_faces.stateChanged.connect(self.update_display_options)
        self.display_options.show_edges.stateChanged.connect(self.update_display_options)
        self.display_options.show_vertices.stateChanged.connect(self.update_display_options)
        self.display_options.edge_width.valueChanged.connect(self.update_display_options)
        self.display_options.vertex_size.valueChanged.connect(self.update_display_options)


    def create_transformation_controls(self, layout):
        transform_layout = QHBoxLayout()

        # Translation controls
        translation_group = QGroupBox("Translation")
        translation_layout = QVBoxLayout()
        translation_group.setLayout(translation_layout)
        self.translate_x = self.create_labeled_slider("X", -5, 5, 0, 0.1, translation_layout)
        self.translate_y = self.create_labeled_slider("Y", -5, 5, 0, 0.1, translation_layout)
        self.translate_z = self.create_labeled_slider("Z", -5, 5, 0, 0.1, translation_layout)
        transform_layout.addWidget(translation_group)

        # Scale control
        scale_group = QGroupBox("Scale")
        scale_layout = QVBoxLayout()
        scale_group.setLayout(scale_layout)
        self.scale_slider = self.create_labeled_slider("Scale", 0.1, 2, 1, 0.1, scale_layout)
        transform_layout.addWidget(scale_group)

        layout.addLayout(transform_layout)

        # Reset button
        self.reset_button = QPushButton("Reset Transformation")
        self.reset_button.clicked.connect(self.reset_transformation)
        layout.addWidget(self.reset_button)

    def create_labeled_slider(self, label, min_val, max_val, initial_val, step, layout):
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel(label))

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val / step))
        slider.setMaximum(int(max_val / step))
        slider.setValue(int(initial_val / step))
        slider.valueChanged.connect(self.apply_transformation)

        value_label = QLabel(f"{initial_val:.1f}")
        slider.valueChanged.connect(lambda v: value_label.setText(f"{v * step:.1f}"))

        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        layout.addLayout(slider_layout)
        return slider

    def load_shape(self):
        shape = self.shape_combo.currentText()
        self.view.clear()

        try:
            if shape == 'Cube':
                self.wireframe = Wireframe.create_cube()
            elif shape == 'Sphere':
                self.wireframe = Wireframe.create_sphere()
            elif shape == 'Tetrahedron':
                self.wireframe = Wireframe.create_tetrahedron()
            elif shape == 'Octahedron':
                self.wireframe = Wireframe.create_octahedron()
            elif shape == 'Icosahedron':
                self.wireframe = Wireframe.create_icosahedron()
            elif shape == 'Torus':
                self.wireframe = Wireframe.create_torus()
            elif shape == 'Cylinder':
                self.wireframe = Wireframe.create_cylinder()
            elif shape == 'Cone':
                self.wireframe = Wireframe.create_cone()

            self.update_display()
            self.reset_transformation()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load shape: {str(e)}")

    def apply_transformation(self):
        if not self.scene.objects:
            return

        try:
            # Apply translation
            translation = np.array([
                self.translate_x.value() * 0.1,
                self.translate_y.value() * 0.1,
                self.translate_z.value() * 0.1
            ])

            # Apply scaling
            scale_factor = self.scale_slider.value() * 0.1
            scaling = np.array([scale_factor, scale_factor, scale_factor])

            for obj in self.scene.objects:
                # Reset to original position and scale
                obj.position = np.zeros(3)
                obj.scale = np.ones(3)

                # Apply new translation and scaling
                obj.position += translation
                obj.scale *= scaling

                obj.apply_transformation()

            self.update_display()
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error applying transformation: {str(e)}")

    def reset_transformation(self):
        for slider in [self.translate_x, self.translate_y, self.translate_z]:
            slider.setValue(0)
        self.scale_slider.setValue(10)  # 1.0 scale

        for obj in self.scene.objects:
            obj.position = np.zeros(3)
            obj.scale = np.ones(3)
            obj.rotation = np.zeros(3)
            obj.apply_transformation()

        self.update_display()

    def show_display_options(self):
        self.display_options.show()

    def change_color(self, component):
        color = QColorDialog.getColor()
        if color.isValid():
            if component == 'face':
                self.display_options.face_color_btn.setColor(color)
            elif component == 'edge':
                self.display_options.edge_color_btn.setColor(color)
            elif component == 'vertex':
                self.display_options.vertex_color_btn.setColor(color)
            self.update_display_options()

    def update_display_options(self):
        for obj in self.scene.objects:
            obj.update_mesh(self.display_options)

    def update_display(self):
        for obj in self.scene.objects:
            obj.update_mesh(self.display_options)


    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "OBJ Files (*.obj);;JSON Files (*.json)")
        if filename:
            try:
                if filename.endswith('.obj'):
                    self.wireframe = Wireframe.load_from_obj(filename)
                elif filename.endswith('.json'):
                    self.wireframe = Wireframe.load_from_json(filename)
                self.update_display()
                self.reset_transformation()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")

    def save_file(self):
        if not self.wireframe:
            QMessageBox.warning(self, "Warning", "No wireframe to save")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "OBJ Files (*.obj);;JSON Files (*.json)")
        if filename:
            try:
                if filename.endswith('.obj'):
                    self.wireframe.save_to_obj(filename)
                elif filename.endswith('.json'):
                    self.wireframe.save_to_json(filename)
                QMessageBox.information(self, "Success", "File saved successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")


    def show_shape_loader(self):
        self.shape_loader.show()
        self.shape_loader.raise_()
        self.shape_loader.activateWindow()

    def on_shape_loaded(self, shape_data):
        print(f"Debug - WireframeViewer: Received shape_data: {shape_data}")
        shape_type, shape_info = shape_data
        try:
            wireframe = self.create_wireframe(shape_type, shape_info)
            if wireframe:
                scene_object = SceneObject(wireframe)
                scene_object.wireframe.name = f"{shape_type}: {shape_info}"  # Set a name for the object
                self.scene.add_object(scene_object)
                scene_object.update_mesh(self.display_options)
                self.view.addItem(scene_object.mesh_item)
                if scene_object.edge_item:
                    self.view.addItem(scene_object.edge_item)
                if scene_object.vertex_item:
                    self.view.addItem(scene_object.vertex_item)
                self.object_list.addItem(scene_object.wireframe.name)
        except Exception as e:
            print(f"Error loading shape: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load shape: {str(e)}")

    def create_wireframe(self, shape_type, shape_info):
        if shape_type == 'basic':
            return Wireframe.create_basic_shape(shape_info)
        elif shape_type == 'parametric':
            return Wireframe.create_parametric_shape(*shape_info)
        elif shape_type == 'fractal':
            return Wireframe.create_fractal(*shape_info)
        elif shape_type == 'text':
            return Wireframe.create_text(shape_info)
        elif shape_type == 'mri':
            return Wireframe.load_mri(shape_info)
        elif shape_type == 'super_resolution':
            return Wireframe.load_super_resolution(shape_info)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

    def remove_selected_object(self):
        selected_items = self.object_list.selectedItems()
        for item in selected_items:
            index = self.object_list.row(item)
            if 0 <= index < len(self.scene.objects):
                scene_object = self.scene.objects[index]

                # Remove items from the view
                if scene_object.mesh_item in self.view.items:
                    self.view.removeItem(scene_object.mesh_item)
                if scene_object.edge_item in self.view.items:
                    self.view.removeItem(scene_object.edge_item)
                if scene_object.vertex_item in self.view.items:
                    self.view.removeItem(scene_object.vertex_item)

                # Remove object from the scene
                self.scene.objects.pop(index)

                # Remove item from the list widget
                self.object_list.takeItem(index)
            else:
                print(f"Invalid index {index} for object removal")

        # Ensure the object_list and scene.objects are in sync
        self.sync_object_list()

    def sync_object_list(self):
        # Clear the object_list
        self.object_list.clear()

        # Repopulate the object_list from scene.objects
        for obj in self.scene.objects:
            # Assuming each object has a name or some identifier
            self.object_list.addItem(f"Object: {obj.wireframe.name if hasattr(obj.wireframe, 'name') else 'Unnamed'}")


    def toggle_animation(self):
        if self.is_playing:
            self.is_playing = False
            self.play_button.setText('Play')
        else:
            self.is_playing = True
            self.play_button.setText('Pause')
            self.last_time = QTime.currentTime()

    def update_scene(self):
        current_time = QTime.currentTime()

        if self.is_playing:
            dt = self.last_time.msecsTo(current_time) / 1000.0
            self.scene.update_physics(dt)

        self.last_time = current_time
        self.scene.apply_transformations()
        self.update_display()


    def load_basic_shape(self, shape_name):
        try:
            self.wireframe = Wireframe.create_basic_shape(shape_name)
            print(f"Debug - WireframeViewer: Basic shape '{shape_name}' created successfully")
        except Exception as e:
            print(f"Debug - WireframeViewer: Error creating basic shape: {str(e)}")
            raise

    def load_parametric_shape(self, shape_info):
        shape_name, u_res, v_res = shape_info
        try:
            self.wireframe = Wireframe.create_parametric_shape(shape_name, u_res, v_res)
            print(f"Debug - WireframeViewer: Parametric shape '{shape_name}' created successfully")
        except Exception as e:
            print(f"Debug - WireframeViewer: Error creating parametric shape: {str(e)}")
            raise

    def load_fractal_shape(self, fractal_info):
        fractal_type, iterations, size = fractal_info
        try:
            if fractal_type == 'Sierpinski Triangle':
                self.wireframe = Wireframe.create_sierpinski_triangle(iterations, size)
            elif fractal_type == 'Koch Snowflake':
                self.wireframe = Wireframe.create_koch_snowflake(iterations, size)
            elif fractal_type == 'Menger Sponge':
                self.wireframe = Wireframe.create_menger_sponge(iterations, size)
            elif fractal_type == 'Apollonian Gasket':
                self.wireframe = Wireframe.create_apollonian_gasket(iterations, size)
            elif fractal_type == 'Jerusalem Cube':
                self.wireframe = Wireframe.create_jerusalem_cube(iterations, size)
            elif fractal_type == 'Sierpinski Tetrahedron':
                self.wireframe = Wireframe.create_sierpinski_tetrahedron(iterations, size)
            elif fractal_type == 'Pythagoras Tree 3D':
                self.wireframe = Wireframe.create_pythagoras_tree_3d(iterations, size)
            else:
                raise ValueError(f"Unknown fractal type: {fractal_type}")
            print(f"Debug - WireframeViewer: Fractal '{fractal_type}' created successfully")
        except Exception as e:
            print(f"Debug - WireframeViewer: Error creating fractal: {str(e)}")
            raise

    def load_text_shape(self, text):
        try:
            self.wireframe = Wireframe.create_text(text)
            print(f"Debug - WireframeViewer: Text shape '{text}' created successfully")
        except Exception as e:
            print(f"Debug - WireframeViewer: Error creating text shape: {str(e)}")
            raise

    def load_mri_data(self, filename):
        try:
            self.wireframe = Wireframe.load_mri(filename)
            print(f"Debug - WireframeViewer: MRI data loaded successfully from {filename}")
        except Exception as e:
            print(f"Debug - WireframeViewer: Error loading MRI data: {str(e)}")
            raise

    def load_super_resolution_data(self, filename):
        try:
            self.wireframe = Wireframe.load_super_resolution(filename)
            print(f"Debug - WireframeViewer: Super-resolution data loaded successfully from {filename}")
        except Exception as e:
            print(f"Debug - WireframeViewer: Error loading super-resolution data: {str(e)}")
            raise



if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = WireframeViewer()
    viewer.show()
    sys.exit(app.exec_())
