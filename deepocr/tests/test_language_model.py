import subprocess

import train_entrypoint


def build_writing(*names):
    def build(command, cwd, capture_output, text, timeout):
        out = command[command.index('--out') + 1]
        for suffix in names:
            open(f'{out}{suffix}', 'w').write(suffix)
        return subprocess.CompletedProcess(command, 0 if names else 1, '', 'lmplz not on PATH')
    return build


def test_a_built_language_model_is_listed_by_role(tmp_path, monkeypatch):
    monkeypatch.setattr(train_entrypoint.subprocess, 'run', build_writing('.bin', '.dict'))
    files, outcome = train_entrypoint.language_model(tmp_path / 'snapshot', tmp_path)
    assert outcome == 'built'
    assert files == [{'role': 'lm', 'path': 'lm/domain.bin'}, {'role': 'lm_dict', 'path': 'lm/domain.dict'}]


def test_a_failed_build_is_reported_and_lists_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_entrypoint.subprocess, 'run', build_writing())
    assert train_entrypoint.language_model(tmp_path / 'snapshot', tmp_path) == ([], 'failed')
    assert 'lmplz not on PATH' in capsys.readouterr().out


def test_a_build_without_the_binary_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(train_entrypoint.subprocess, 'run', build_writing('.dict'))
    assert train_entrypoint.language_model(tmp_path / 'snapshot', tmp_path) == ([], 'failed')


def test_a_run_on_the_builtin_base_fine_tunes_the_builtin_and_keeps_its_language_model(tmp_path, monkeypatch):
    builtin = tmp_path / 'builtin'
    builtin.mkdir()
    for name in ('deepocr-v3.pt', 'union_bytes_v3.bin', 'union_bytes_v3.dict'):
        (builtin / name).write_text(name)
    monkeypatch.setattr(train_entrypoint, 'BUILTIN_DIR', builtin)
    base, output = tmp_path / 'base', tmp_path / 'out'
    base.mkdir(), output.mkdir()

    assert train_entrypoint.base_env(base) == {}
    (base / 'builtin').touch()
    assert train_entrypoint.base_env(base) == {'TRAIN_BASE_CHECKPOINT': str(builtin / 'deepocr-v3.pt')}

    assert train_entrypoint.builtin_language_model(output) == [
        {'role': 'lm', 'path': 'lm/builtin.bin'}, {'role': 'lm_dict', 'path': 'lm/builtin.dict'}]
    assert (output / 'lm' / 'builtin.dict').read_text() == 'union_bytes_v3.dict'
