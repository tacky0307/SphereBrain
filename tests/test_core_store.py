import sqlite3

from sphere_brain.core import Store


def test_setup_reuses_descriptors_but_creates_a_session(tmp_path):
    database = tmp_path / "brain.sqlite"
    store = Store(database)
    arguments = {
        "config_version": "v1",
        "configuration": {"learning_rate": 0.2},
        "structure_version": "sphere-v1",
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [("a", "b")],
    }

    first = store.setup(**arguments)
    second = store.setup(**arguments)

    assert first.project_id == second.project_id
    assert first.experiment_id == second.experiment_id
    assert first.configuration_id == second.configuration_id
    assert first.structure_id == second.structure_id
    assert first.session_id != second.session_id

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM projects").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM experiments").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM configurations").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM structures").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM nodes").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM edges").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM sessions").fetchone()[0] == 2


def test_changed_configuration_gets_a_new_configuration_only(tmp_path):
    store = Store(tmp_path / "brain.sqlite")
    common = {"config_version": "v1", "structure_version": "sphere-v1"}

    first = store.setup(configuration={"rate": 1}, **common)
    second = store.setup(configuration={"rate": 2}, **common)

    assert first.configuration_id != second.configuration_id
    assert first.structure_id == second.structure_id
