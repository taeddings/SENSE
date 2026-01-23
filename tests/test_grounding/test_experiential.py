import pytest
import os
from sense_v2.grounding.experiential import ExperientialGrounding, GroundingResult

class TestExperientialGrounding:
    
    @pytest.fixture
    def grounding(self):
        return ExperientialGrounding(use_tools=False)
    
    def test_verify_file_exists(self, grounding):
        # Create a temporary file
        temp_path = '/tmp/test_sense_experiential.txt'
        with open(temp_path, 'w') as f:
            f.write('test content')
        
        result = grounding.verify_file_exists(temp_path)
        assert result.verified is True
        assert 'exists' in result.explanation.lower()
        assert result.confidence == 1.0
        
        # Clean up
        os.remove(temp_path)
        
        # Verify non-existent file
        result_nonexistent = grounding.verify_file_exists(temp_path)
        assert result_nonexistent.verified is False
        assert 'does not exist' in result_nonexistent.explanation.lower()
        assert result_nonexistent.confidence == 0.8
    
    def test_verify_command_succeeded(self, grounding):
        # Mock successful command result
        success_result = {
            'returncode': 0,
            'stdout': 'command output',
            'stderr': ''
        }
        result = grounding.verify_command_succeeded('echo success', success_result)
        assert result.verified is True
        assert result.confidence == 1.0
        assert 'succeeded: True' in result.explanation
        
        # Mock failed command
        fail_result = {
            'returncode': 1,
            'stdout': '',
            'stderr': 'error message'
        }
        result_fail = grounding.verify_command_succeeded('bad command', fail_result)
        assert result_fail.verified is False
        assert result_fail.confidence < 1.0
        assert 'succeeded: False' in result_fail.explanation
    
    def test_verify_state_change(self, grounding):
        # Record initial state
        initial_state = {'counter': 0, 'status': 'idle'}
        grounding.record_state(initial_state)
        
        # Expected change
        expected_change = {
            'counter': {'operator': 'eq', 'value': 1},
            'status': {'operator': 'eq', 'value': 'active'}
        }
        current_state = {'counter': 1, 'status': 'active'}
        
        result = grounding.verify_state_change(expected_change, current_state)
        assert result.verified is True
        assert result.confidence == 1.0
        assert 'counter' in result.explanation
        assert 'status' in result.explanation
        
        # Failed change
        failed_change = {
            'counter': {'operator': 'eq', 'value': 2}
        }
        result_fail = grounding.verify_state_change(failed_change, current_state)
        assert result_fail.verified is False
        assert result_fail.confidence == 0.7
    
    def test_verify_action_outcome(self, grounding):
        # File create action
        temp_path = '/tmp/test_action.txt'
        with open(temp_path, 'w') as f:
            f.write('test')
        result_file = grounding.verify_action_outcome('file_create', {'path': temp_path}, None)
        assert result_file.verified is True
        os.remove(temp_path)
        
        # Command exec
        success_cmd = {'returncode': 0}
        result_cmd = grounding.verify_action_outcome('command_exec', {'command': 'ls'}, success_cmd)
        assert result_cmd.verified is True
        
        # Unknown action
        result_unknown = grounding.verify_action_outcome('unknown', {}, None)
        assert result_unknown.verified is False
        assert 'Unknown action type' in result_unknown.explanation
    
    def test_no_previous_state(self, grounding):
        expected_change = {'key': 1}
        result = grounding.verify_state_change(expected_change)
        assert result.verified is False
        assert 'No previous state recorded' in result.explanation
        assert result.confidence == 0.5