"""A wrapper for some basic PyQt functions

Created 2019, Contributors: Nathan
"""

import functools
import logging
import sys
import time
from pathlib import Path
from typing import Union, cast, Optional
import yaml

from PyQt5 import QtWidgets, QtCore, QtGui

from lib.configy import Configy
from lib import print_colour

log = logging.getLogger(__name__)


# Functions ########
def check_app_exists():
    """
    QApplication is intended to be a singleton, so we should check to see if it exists before trying to create another
    QApplication runs the main event loop that Qt programs are driven by
    """

    app = QtWidgets.QApplication.instance()
    if app is None:
        app_existed = False
        app = QtWidgets.QApplication(sys.argv)
    else:
        app_existed = True

    return app, app_existed


def restore_arrow_cursor():
    """Restore cursor to arrow.
    Useful in limited circumstances for modules called from a pyqt script, where the module needs to interrupt
    the pyqt script (like with a popup or input dialog) when the pyqt script has already set the cursor to "Waiting"
    """
    if QtWidgets.QApplication.instance():
        QtWidgets.qApp.setOverrideCursor(QtCore.Qt.ArrowCursor)
        QtCore.QCoreApplication.processEvents()


def restore_override_cursor():
    """Restore to previous cursor"""
    if QtWidgets.QApplication.instance():
        QtWidgets.qApp.restoreOverrideCursor()
        QtCore.QCoreApplication.processEvents()


# Decorators ########
def cursor_switcher(func):

    """Decorator that changes the cursor to "Waiting" then back to "Arrow" during a function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        QtWidgets.qApp.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtCore.QCoreApplication.processEvents()

        try:
            value = func(*args, **kwargs)
        except TypeError:
            # A lame fix to account for the fact that pyqt adds an argument to a function when it is used as a callback
            value = func(*args[:-1], **kwargs)
        finally:
            QtWidgets.qApp.setOverrideCursor(QtCore.Qt.ArrowCursor)
            QtCore.QCoreApplication.processEvents()

        return value
    return wrapper


# Classes ########
class Dialogs(Configy):
    """A collection of PyQt dialogs, with configy used to saved the values from them for re-populating next time"""

    def __init__(self, identifier: str):
        """
        Configy will be used to remember/auto-populate dialog input
        :param identifier: A unique string identifier. Recommended -> __file__ + 'some descriptor'
        """
        super().__init__(identifier)

        self.app, self.app_existed = check_app_exists()

    def set_new_identifier(self, new_identifier: str):
        """If more than method needs to be run on the same Dialogs instance, the identifier
        should be changed first, so that the saved fields don't get overwritten
        """
        super().__init__(new_identifier)

    def select_save_path(self, accepted_file_types: str = '*.gif') -> Optional[Path]:
        """Open dialog to choose a path for saving a file"""

        dialog = QtWidgets.QFileDialog()
        selected_path, ok = dialog.getSaveFileName(caption='Select Save Path',
                                                   directory=self.fields[0],
                                                   filter=accepted_file_types)
        if not ok:
            if self.app_existed:
                return None
            else:
                raise SystemExit('No file selected')

        self.save_fields(selected_path)

        return Path(selected_path)

    def select_files_dialog(self, accepted_file_types: str = '*.csv') -> Optional[list[Path]]:
        """Open multi file selection dialog"""

        dialog = QtWidgets.QFileDialog()
        selected_paths, ok = dialog.getOpenFileNames(caption='Select File(s)',
                                                     directory=str(self.fields[0]),
                                                     filter=accepted_file_types)
        if not ok:
            if self.app_existed:
                return None
            else:
                raise SystemExit('No file selected')

        selected_paths = list(map(Path, selected_paths))

        self.save_fields(f'{selected_paths[0].parent}/' + ' '.join(f'"{pth.name}"' for pth in selected_paths))

        return selected_paths

    def select_file_dialog(self, accepted_file_types: str = '*.csv') -> Optional[Path]:
        """Open single file selection dialog"""

        dialog = QtWidgets.QFileDialog()
        selected_path, ok = dialog.getOpenFileName(caption='Select File',
                                                   directory=str(self.fields[0]),
                                                   filter=accepted_file_types)
        if not ok:
            if self.app_existed:
                return None
            else:
                raise SystemExit('No file selected')

        self.save_fields(selected_path)

        return Path(selected_path)

    def select_folder_dialog(self) -> Optional[Path]:
        """Open folder select dialog"""

        dialog = QtWidgets.QFileDialog()
        selected_path = dialog.getExistingDirectory(caption='Select Folder',
                                                    directory=str(self.fields[0]))
        if not selected_path:
            if self.app_existed:
                return None
            else:
                raise SystemExit('No file selected')

        self.save_fields(selected_path)

        return Path(selected_path)

    def get_text_dialog(self, dlg_title: str, prompt: str) -> Optional[str]:
        """Open text input dialog"""

        print('\nWaiting for text input..')  # To give the user a clue in case the dialog is behind something

        dlg = QtWidgets.QInputDialog()
        text_input, ok = QtWidgets.QInputDialog.getText(dlg, dlg_title, prompt, QtWidgets.QLineEdit.Normal,
                                                        str(self.fields[0]))
        if not ok:
            if self.app_existed:
                return None
            else:
                raise SystemExit('No file selected')

        self.save_fields(text_input)

        print('Input Recorded')

        return text_input

    def load_yamls(self) -> tuple[Optional[list[dict]], Optional[list[Path]]]:
        """Load one or more yaml config files into dictionaries"""

        config_paths = self.select_files_dialog(accepted_file_types='Config File (*.yaml)')

        config_dicts = [self._parse_yaml(path) for path in config_paths] if config_paths else None

        return config_dicts, config_paths

    def load_config_get_tag(self, request_tag: bool = True) -> tuple[dict, str, Path]:
        """A one-stop fn for for YAML config loading and tag input dialog"""

        config_dicts, config_paths = self.load_yamls()

        if request_tag:
            self.set_new_identifier('pyqt_helper load yaml get tag')
            tag = self.get_text_dialog(dlg_title='Enter Tag', prompt='Provide a description tag for file/folder name')
            self.save_fields([tag])
        else:
            tag = ''

        return config_dicts[0], tag, config_paths[0].parent

    def _parse_yaml(self, path: Path) -> dict:
        """Load a yaml file into a dictionary"""

        try:
            with path.open('r', newline='', encoding='utf8') as f:
                config_dict = yaml.unsafe_load(f)
            print(f'\nImported config file:\n{path.name}')
        except IOError:
            raise SystemExit('No YAML Loaded')
        except Exception:
            print_colour.printy("\nLooks like there's a problem with your YAML :(", 're_raise_yellow')
            raise

        self._replace_special_chars(config_dict)

        return config_dict

    @staticmethod
    def _replace_special_chars(config: dict):
        """Allow fancy characters to be added in to parsed yaml"""

        special_chars = {'Very_Unique_Key_for_UM': r'$\mu$m',
                         'Very_Unique_Key_for_dot': u'\u2219',
                         '\\n': '\n'}

        if 'plot_the_data' in config.keys():
            for line, data in config['plot_the_data'].items():
                for key, val in data.items():

                    if isinstance(val, list) and isinstance(val[0], str):
                        for unique_key, char in special_chars.items():
                            val = [f'{item}'.replace(unique_key, char) for item in val]
                            config['plot_the_data'][line][key] = val

                    elif isinstance(val, str):
                        for unique_key, char in special_chars.items():
                            val = val.replace(unique_key, char)
                            config['plot_the_data'][line][key] = val

    def _dig_through_dicts_and_lists(self, nested_object, keys_or_indexes=None):
        if isinstance(nested_object, dict):
            for key, val in nested_object.items():
                self._dig_through_dicts_and_lists(val)
        elif isinstance(nested_object, list):
            for index, item in enumerate(nested_object):
                self._dig_through_dicts_and_lists(item)
        elif isinstance(nested_object, str) and nested_object.lower() == 'none':
            nested_object = None


class MultiDialog(Configy, QtWidgets.QDialog):
    """An expandable input dialog using PyQt
    - Pass a list of input prompts and default values
    - User inputs are stored in an instance variable and get be accessed after the dialog is closed
    """

    def __init__(self,
                 identifier: str,  # A unique string identifier. Recommended -> __file__ + 'some descriptor'
                 input_prompts: list[Union[str, tuple[str, list]]],  # See notes below
                 default_val_overrides: list[str] = None,
                 dialog_title: str = 'Inputs',
                 min_dlg_width: int = None,
                 unstringify_outputs: bool = True,
                 modal: bool = False):
        """input_prompts - If a value is a string, a LineEdit will be used
                         - If a value is a tuple[str], a TextEdit will be used
                         - If a value is a tuple[str, List], a drop-down selector will be used
        """

        app, self.app_existed = check_app_exists()

        Configy.__init__(self, identifier)
        QtWidgets.QDialog.__init__(self)

        self.closeEvent = self.dialog_close_event
        self.okay_or_enter = False
        self.unstringify_outputs = unstringify_outputs
        self.failed_to_convert_strings = False

        # Setup window
        self.setWindowTitle(dialog_title)
        self.setModal(modal)
        if min_dlg_width:
            self.setMinimumWidth(min_dlg_width)

        # Choose default vals
        if default_val_overrides:
            self.fields = [str(x) for x in default_val_overrides]
        else:
            # Ensure there are enough Configy fields for prompts
            self.fields = [''] * len(input_prompts) if len(self.fields) < len(input_prompts) else self.fields

        # Add rows of input elements
        layout = QtWidgets.QFormLayout()
        self.ui_elements = []
        for prompt, default in zip(input_prompts, self.fields):

            # Determine prompt type
            if type(prompt) == str:
                ui_element = QtWidgets.QLineEdit(default, self)
            elif type(prompt) == tuple and len(prompt) == 1:
                prompt = prompt[0]
                ui_element = QtWidgets.QTextEdit(default, self)
            elif type(prompt) == tuple:
                prompt, selection_list = prompt
                ui_element = QtWidgets.QComboBox(self)
                ui_element.addItems([str(x) for x in selection_list])
                ui_element.setCurrentText(default)
            else:
                raise Exception('Unsupported prompt type')

            self.ui_elements.append(ui_element)
            layout.addRow(QtWidgets.QLabel(prompt), ui_element)

        # Add "Okay" and "Cancel" button
        self.btn_canc = QtWidgets.QPushButton('Cancel', self)
        self.btn_okay = QtWidgets.QPushButton('Okay', self)
        self.btn_canc.clicked.connect(self.cancel_fun)
        self.btn_okay.clicked.connect(self.okay_fun)
        layout.addRow(self.btn_okay, self.btn_canc)

        # Apply layout and show dialog
        self.setLayout(layout)

        self.show()
        self.exec()

    def unstringify(self):
        """Take user values and try and convert items that don't look like strings"""

        try:
            new_fields = list()
            for x in self.fields:
                if x.strip('-').isdigit():
                    x = int(x)
                elif x.strip('-').replace('.', '').isdigit():
                    x = float(x)
                elif x.lower() == 'true':
                    x = True
                elif x.lower() == 'false':
                    x = False
                else:
                    x = x
                new_fields.append(x)

            self.fields = new_fields

        except Exception as err:
            msg = f'PyQt Helper failed to convert strings: {err}'
            print(msg)
            log.info(msg)
            self.failed_to_convert_strings = err

    def cancel_fun(self):
        """User presses Cancel button, or X"""
        if self.app_existed:
            self.close()
        else:
            raise SystemExit

    def okay_fun(self):
        """User presses okay button, or Enter"""

        # Store user input
        self.fields = [element.currentText() if isinstance(element, QtWidgets.QComboBox) else
                       element.toPlainText() if isinstance(element, QtWidgets.QTextEdit) else
                       element.text()
                       for element in self.ui_elements]

        # Save input with Configy
        self.save_fields()
        self.okay_or_enter = True

        if self.unstringify_outputs:
            self.unstringify()

        self.close()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter:
            self.okay_fun()
        if event.key() == QtCore.Qt.Key_Escape:
            self.cancel_fun()

    def dialog_close_event(self, event: QtCore.QEvent) -> None:
        """Triggered by Cancel, "X", or Escape"""
        if not self.okay_or_enter:
            self.fields = None
        event.accept()


class MessageBox:
    """A message box dialog to alert the user of something"""

    def __init__(self, text_1: str, text_2: str, title: str, window_type: str, pause_before_close: float = None):

        if QtWidgets.QApplication.instance():
            rely_on_qt_event_loop_to_run_timer = True
        else:
            rely_on_qt_event_loop_to_run_timer = False

        self.app, app_existed = check_app_exists()

        self.msg = QtWidgets.QMessageBox()

        if 'question' in window_type.lower():
            self.msg.setIcon(QtWidgets.QMessageBox.Question)
        elif 'info' in window_type.lower():
            self.msg.setIcon(QtWidgets.QMessageBox.Information)
        elif 'warn' in window_type.lower():
            self.msg.setIcon(QtWidgets.QMessageBox.Warning)
        elif 'critical' in window_type.lower():
            self.msg.setIcon(QtWidgets.QMessageBox.Critical)

        self.msg.setText(text_1)
        self.msg.setInformativeText(text_2)
        self.msg.setWindowTitle(title)
        self.msg.show()

        # Open and start kill timer, or open and block
        if pause_before_close:
            self.app.processEvents()  # Ensure dialog contents are visible (text, button, etc)
            if rely_on_qt_event_loop_to_run_timer:
                self.thready = TimerThread(pause_before_close)
                self.thready.timer_done.connect(self.kill)
                self.thready.start()
            else:
                # This is relevant for non-gui scripts
                # Without a QT GUI in the background we can't use a QT timer to close the dialog box
                time.sleep(pause_before_close)
        else:
            self.app.exec()

    def kill(self):
        self.msg.close()
        self.thready.quit()


class TimerThread(QtCore.QThread):
    timer_done = QtCore.pyqtSignal()

    def __init__(self, pause_before_close_s):
        super().__init__()
        self.pause_before_close_s = pause_before_close_s

    def run(self):
        timer = QtCore.QTimer(self)
        timer.singleShot(int(self.pause_before_close_s * 1000), self.timer_done.emit)
        loop = QtCore.QEventLoop()
        loop.exec()


class FilesAndFoldersDialog(QtWidgets.QFileDialog):
    """A PyQt dialog for selecting BOTH files and folders doesn't exist.
    This is a bit of a hack to make it work
    """

    def __init__(self, name_filter=None):

        app, app_existed = check_app_exists()

        super().__init__()

        self.selected_files = None
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.ExistingFiles)
        self.setNameFilter(name_filter)
        buttons = cast(list[QtWidgets.QPushButton], self.findChildren(QtWidgets.QPushButton))
        self.openBtn = next(x for x in buttons if 'open' in str(x.text()).lower())
        self.openBtn.clicked.disconnect()
        self.openBtn.clicked.connect(self.open_clicked)
        self.tree = cast(QtWidgets.QTreeView, self.findChild(QtWidgets.QTreeView))
        self.show()
        app.exec()

    def open_clicked(self):
        indexes = self.tree.selectionModel().selectedIndexes()
        files = []
        for i in indexes:
            if i.column() == 0:
                files.append(self.directory().absolutePath() + '/' + i.data())
        self.selected_files = files
        self.close()


if __name__ == '__main__':

    the_dlg = MultiDialog(__file__ + 'test', ['Prompt 1', ('Prompt 2', ), ('Prompt 3', [1, 2, 3])])
    print(the_dlg.fields)
