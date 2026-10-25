"""Final eight-run design and unchanged-production safety regressions (offline)."""

import copy
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import agent, claim_verifier, evidence_checker, writer
from app.knowledge import load_knowledge_base
from evaluations.reflection_ab import core, worker


FACTS = next(doc['facts'] for doc in load_knowledge_base() if doc['id'] == 'CDC001')
GOOD = ''.join(FACTS) + '[1]'
BAD_TREATMENT = '睡前喝牛奶能够治愈失眠。[1]'
BAD_PARAMETER = '资料未提及睡前拉伸20分钟的效果，因此无法确认睡前拉伸2.0分钟的效果。[1]'
OMITTED = '资料不足，无法确认该拉伸方案的效果。[1]'


def case(case_id):
    return next(item for item in core.load_cases()['cases'] if item['id'] == case_id)


class Model:
    def __init__(self, initial=GOOD, repair=GOOD, fail_stage=None, truncate=False):
        self.initial, self.repair = initial, repair
        self.calls = []
        self.fail_stage, self.truncate = fail_stage, truncate

    def __call__(self, *args, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        is_writer = 'article' in kwargs['format']['properties']
        stage = 'Writer' if is_writer else 'Verifier'
        if stage == self.fail_stage:
            raise TimeoutError('fixed offline request timeout')
        if is_writer:
            repair = '上一版文章' in kwargs['messages'][-1]['content']
            text = self.repair if repair else self.initial
            content = json.dumps({'article': text}, ensure_ascii=False)
        else:
            claim = kwargs['messages'][-1]['content'].split('待核查主张：', 1)[1]
            label = 'not_supported' if '牛奶' in claim else 'supported'
            content = json.dumps({'verdict': label, 'reason': 'offline fixed response',
                                  'evidence_id': 0 if label == 'not_supported' else 1})
        return SimpleNamespace(message=SimpleNamespace(content=content, thinking=None),
                               done=True, done_reason='length' if self.truncate else 'stop',
                               eval_count=12, prompt_eval_count=40)


def test_production_configuration_capture_is_offline_and_keeps_verifier_separate():
    config = core.production_config()
    assert config['writer']['options'] == writer.OPTIONS == {
        'temperature': 0, 'seed': 42, 'num_predict': 512, 'num_ctx': 4096}
    assert config['writer']['protocol'] == 'json_article_v1'
    assert config['writer']['request_timeout_seconds'] == config['verifier']['request_timeout_seconds'] == 45
    assert config['verifier']['options'] == {'temperature': 0}
    assert config['verifier']['format'] == claim_verifier.CHECK_SCHEMA
    assert config['verifier']['configuration_source'].endswith('offline_capture')


def test_two_runs_have_disjoint_calls_equal_initial_requests_and_no_shared_prefix():
    model = Model(BAD_PARAMETER, OMITTED)
    pair = core.run_pair(case('R-INSUFFICIENT'), model)
    a, b = pair['rows']['A'], pair['rows']['B']
    assert pair['design'] == 'independent_end_to_end_paired'
    assert pair['shared_model_call_ids'] == []
    assert a['end_to_end_invocation_count'] == b['end_to_end_invocation_count'] == 1
    assert not a['public_production_entrypoint_called'] and not b['public_production_entrypoint_called']
    assert set(a['model_call_ids']).isdisjoint(b['model_call_ids'])
    assert a['model_calls'][0]['request'] == b['model_calls'][0]['request']
    assert a['production_config'] == b['production_config']
    assert len(model.calls) == pair['physical_model_call_count'] == 3
    assert a['status'] == 'evaluation_failed' and a['revision_count'] == 0
    assert b['status'] == 'completed' and b['revision_count'] == 1
    assert a['gate_outcomes']['task_parameter_fidelity_passed'] is False
    assert b['gate_outcomes']['task_parameter_fidelity_passed'] is True
    assert all(row['publication_integrity_passed'] for row in (a, b))


def test_no_rewriter_or_safety_setting_is_removed_from_off():
    a = core.run_case(case('R-INJECT'), 'A', Model())
    assert a['status'] == 'evaluation_failed' and a['revision_count'] == 0
    assert not any(stage['stage'] == 'Reflection' for stage in a['stages'])
    assert any('牛奶' in item['claim'] for item in a['check_history'][0]['claims'])
    assert a['gate_outcomes']['medical_evidence_passed'] is False
    assert a['answer'] == writer.SAFE_STOP and '牛奶' not in a['answer']


def test_controlled_injection_happens_once_each_and_never_in_repair(monkeypatch):
    monkeypatch.setenv('INJECT_BAD_FACT', '1')
    monkeypatch.setenv('INJECT_FALSE_CLAIM', '1')
    model = Model()
    pair = core.run_pair(case('R-INJECT'), model)
    for row in pair['rows'].values():
        assert len(row['fault_applications']) == 1
        assert row['fault_applications'][0]['applied'] is True
        assert row['initial_draft'] == GOOD + '\n' + BAD_TREATMENT
        assert row['initial_draft'].count(BAD_TREATMENT) == 1
    b = pair['rows']['B']
    assert b['status'] == 'completed' and b['draft'] == GOOD
    assert len(b['check_history']) == 2 and b['reflection_repair_succeeded']
    assert all(stage['output']['draft'] == GOOD for stage in b['stages']
               if stage['stage'] == 'Writer' and stage['pass'] == 1)
    assert len([request for request in model.calls if 'article' in request['format']['properties']]) == 3
    assert __import__('os').environ['INJECT_FALSE_CLAIM'] == '1'


@pytest.mark.parametrize('strategy', ['A', 'B'])
def test_no_evidence_and_content_insufficiency_keep_production_terminal_meanings(strategy):
    never = Mock(side_effect=AssertionError('Terminal Retriever must not invoke model'))
    no_evidence = core.run_case(case('R-NONE'), strategy, never)
    assert no_evidence['status'] == 'no_evidence' and no_evidence['no_evidence_safe_stop']
    assert no_evidence['model_call_count'] == no_evidence['revision_count'] == 0
    long = {**case('R-NORMAL'), 'task': '请依据睡眠资料写2000字长篇科普。'}
    insufficient = core.run_case(long, strategy, never)
    assert insufficient['status'] == 'insufficient_content'
    assert insufficient['model_call_count'] == insufficient['revision_count'] == 0
    assert insufficient['answer'] == agent.retriever_node(agent.HealthState(
        task=long['task'], evidence=[], draft='', issues=[], revision_count=0, answer='',
        status='running', writer_runs=[], generation_failed=False, checker_details={}))['answer']
    assert insufficient['publication_integrity_passed']
    never.assert_not_called()


def test_on_matches_actual_production_graph_state_and_request_sequence(monkeypatch):
    initial = GOOD.replace('60岁', '6:0岁')
    model_eval = Model(initial)
    row = core.run_case(case('R-NORMAL'), 'B', model_eval)
    model_prod = Model(initial)
    monkeypatch.setenv('INJECT_BAD_FACT', '0')
    monkeypatch.setenv('INJECT_FALSE_CLAIM', '0')
    monkeypatch.setattr(agent, 'chat', model_prod)
    monkeypatch.setattr(claim_verifier, 'chat', model_prod)
    actual = agent.run_health_agent(case('R-NORMAL')['task'])
    assert row['status'] == actual['status'] == 'completed'
    assert row['answer'] == actual['answer']
    assert row['revision_count'] == actual['revision_count'] == 1
    assert len(row['evidence']) == actual['evidence_count']
    assert model_eval.calls == model_prod.calls
    assert row['production_route_identity_verified']
    assert all(proof['compiled_current_source_equals_loaded'] for proof in row['runtime_source_proof'])


def test_model_options_requests_and_raw_responses_are_not_rewritten():
    model = Model()
    row = core.run_case(case('R-INJECT'), 'B', model)
    assert [call['request'] for call in row['model_calls']] == model.calls
    for call in row['model_calls']:
        assert call['response_content'] == call['response']['message']['content']
        assert call['options'] == (writer.OPTIONS if call['stage'] == 'Writer' else {'temperature': 0})
        assert call['request_sha256'] == core.digest(call['request'])
        assert call['completed'] and call['elapsed_seconds'] >= 0
    assert row['final_checker_completed'] and row['gate_outcomes']['checker_passed']


@pytest.mark.parametrize('fail_stage', ['Writer', 'Verifier'])
def test_request_timeout_keeps_safe_production_status_and_diagnostic(fail_stage):
    row = core.run_case(case('R-INJECT'), 'B', Model(fail_stage=fail_stage))
    assert row['status'] == 'evaluation_failed' and row['answer'] == writer.SAFE_STOP
    assert row['error']['type'] == 'TimeoutError'
    assert row['revision_count'] == 0 and row['reflection_runs'] == 0
    assert row['publication_integrity_passed'] and not row['final_checker_completed']
    assert row['model_calls'][-1]['error']['type'] == 'TimeoutError'
    assert row['stages'][-1]['stage'] == 'Stop'


def test_truncation_never_reaches_checker_or_publishes_partial_json():
    row = core.run_case(case('R-NORMAL'), 'B', Model(truncate=True))
    assert row['status'] == 'evaluation_failed' and row['answer'] == writer.SAFE_STOP
    assert row['check_history'] == [] and row['revision_count'] == 0
    assert row['writer_runs'][0]['truncated'] is True
    assert row['model_calls'][0]['truncated'] is True
    assert row['final_task_parameter_fidelity'] is None


def test_repair_failure_does_not_reuse_the_initial_gate_as_final():
    class RepairTimeout(Model):
        def __call__(self, *args, **kwargs):
            if '上一版文章' in kwargs['messages'][-1]['content']:
                self.calls.append(copy.deepcopy(kwargs))
                raise TimeoutError('repair request timeout')
            return super().__call__(*args, **kwargs)
    row = core.run_case(case('R-INJECT'), 'B', RepairTimeout())
    assert row['status'] == 'evaluation_failed' and row['revision_count'] == 1
    assert len(row['check_history']) == 1
    assert row['final_task_parameter_fidelity'] is None
    assert row['gate_outcomes']['task_parameter_fidelity_passed'] is None
    assert not row['final_checker_completed']
    assert not any(stage['stage'] == 'Verifier' and stage['pass'] == 1 for stage in row['stages'])


def test_repeated_parameter_drift_stops_after_one_rewrite():
    row = core.run_case(case('R-INSUFFICIENT'), 'B', Model(BAD_PARAMETER, BAD_PARAMETER))
    assert row['status'] == 'evaluation_failed' and row['revision_count'] == 1
    assert row['model_call_count'] == 2
    assert row['final_task_parameter_fidelity']['passed'] is False
    assert row['check_history'][-1]['checker_details']['complete']
    assert row['publication_integrity_passed']


def test_parameter_gate_error_stops_without_extra_rewrite(monkeypatch):
    monkeypatch.setattr(evidence_checker, 'check_task_parameter_fidelity',
                        Mock(side_effect=RuntimeError('fixed offline gate exception')))
    row = core.run_case(case('R-INSUFFICIENT'), 'B', Model(OMITTED))
    assert row['status'] == 'evaluation_failed' and row['revision_count'] == 0
    assert row['check_history'][0]['checker_details']['complete'] is False
    assert row['final_task_parameter_fidelity']['passed'] is False
    assert row['publication_integrity_passed']


@pytest.mark.parametrize('timeout', [90, 120, 0, True])
def test_worker_rejects_timeout_override_before_creating_client(monkeypatch, tmp_path, timeout):
    request = {'case': case('R-NONE'), 'strategy': 'A', 'request_timeout_seconds': timeout,
               'event_path': str(tmp_path / 'events.jsonl')}
    monkeypatch.setattr(worker.sys, 'stdin', io.StringIO(json.dumps(request)))
    client = Mock(side_effect=AssertionError('Invalid worker request must not create client'))
    monkeypatch.setattr(worker, 'Client', client)
    with pytest.raises(ValueError, match='production timeout'):
        worker.main()
    client.assert_not_called()
