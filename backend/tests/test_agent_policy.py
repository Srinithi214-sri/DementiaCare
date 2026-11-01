from agent.policy import PolicyEngine

def test_policy_allowed_action():
    assert PolicyEngine.enforce("greeting", "greeting", "speak") is True

def test_policy_rejected_action():
    assert PolicyEngine.enforce("greeting", "greeting", "launch_drone") is False

def test_policy_rejected_response_type():
    assert PolicyEngine.enforce("greeting", "escalation", "speak") is False

def test_policy_unknown_intent():
    assert PolicyEngine.enforce("unknown_intent", "greeting", "speak") is False

def test_policy_safety_escalation():
    assert PolicyEngine.enforce("safety_escalation", "escalation", "none") is True
    assert PolicyEngine.enforce("safety_escalation", "escalation", "notify_caregiver") is True
    assert PolicyEngine.enforce("safety_escalation", "reminiscence", "speak") is False
