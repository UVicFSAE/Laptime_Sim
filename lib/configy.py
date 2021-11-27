"""Loads and saves list of objects to JSON file

Created 2019, Contributors: npaolini
"""

import json
from typing import Union, Any
from pathlib import Path
import pytest


class Configy:

    _json_path = Path.home() / 'saved_fields_rev12.json'

    def __init__(self, identifier: str):
        """Each instance of Configy, with it's unique identifier, can be used to save and load a list of objects

        :param identifier: Any string, to be used as a unique identifier. Recommended: __file__ + 'descriptor string'
        """

        blank = [''] * 99
        self._id = Path(identifier).name
        self._json = json.load(self._json_path.open('r')) if self._json_path.exists() else {self._id: blank}
        self.fields = self._json[self._id] if self._id in self._json else blank

    def save_fields(self, fields: Union[Any, list[Any]] = None) -> None:
        """Save fields to JSON file"""

        self.fields = [fields] if fields and type(fields) != list else fields or self.fields
        self._json[self._id] = self.fields
        try:
            json.dump(self._json, self._json_path.open('w'))
        except PermissionError:
            print('\nPermission error, Configy could not save to JSON file')


# noinspection PyProtectedMember
class Tests:

    test_name = __file__ + 'unit test'

    @pytest.fixture()
    def config_existing_file(self):
        Configy._json_path = Path.home() / 'existing file.json'
        config = Configy(self.test_name)
        config.save_fields('')
        return Configy(self.test_name)

    @pytest.fixture()
    def config_non_existing_file(self):
        Configy._json_path = Path.home() / 'non existent file'
        return Configy(self.test_name)

    def test_new_json_data_is_dict(self, config_non_existing_file):
        assert type(config_non_existing_file._json) == dict, 'json_data not dict'

    def test_id_in_new_json_dict(self, config_non_existing_file):
        assert config_non_existing_file._id in config_non_existing_file._json, 'identifier missing from json_dict'

    def test_new_field_is_string_list(self, config_non_existing_file):
        config = config_non_existing_file
        assert len(config.fields) > 1 and all(type(x) == str for x in config.fields), 'fields is not list of str'

    def test_save_str(self, config_existing_file):
        config_existing_file.save_fields('test_field')
        assert Configy(self.test_name).fields[0] == 'test_field', 'String not saved correctly'

    def test_save_list(self, config_existing_file):
        config_existing_file.save_fields([1, 2, 3])
        assert Configy(self.test_name).fields == [1, 2, 3], 'List not saved correctly'


if __name__ == '__main__':

    config_loader = Configy(__file__ + 'test')
    print('\n', config_loader.fields, '\n')
    config_loader.save_fields(['X', 'Y', {'Z': 1}])

    pytest.main([__file__])
