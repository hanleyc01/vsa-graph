./.venv/bin/pytest ./vsa_graph/tests
./.venv/bin/black ./vsa_graph -l79
./.venv/bin/mypy ./vsa_graph --strict