from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from kg_gen.steps import _1_get_entities as entities_step
from kg_gen.steps import _2_get_relations as relations_step


def response(content, finish_reason):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason)])


class HostedVllmDiagnosticsTests(TestCase):
    def test_entity_length_is_recorded_and_rejected(self):
        with patch.object(entities_step.litellm, "completion",
                          return_value=response('{"entities": ["robot"]}', "length")):
            captured = []
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                entities_step.get_entities(
                    "A robot moves.", use_litellm_prompt=True,
                    model="hosted_vllm/example",
                    response_hook=lambda *args: captured.append(args))
            self.assertEqual(captured, [('{"entities": ["robot"]}', "length")])

    def test_relation_length_is_recorded_and_rejected(self):
        with patch.object(relations_step.litellm, "completion",
                          return_value=response('{"relations": []}', "length")):
            captured = []
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                relations_step.get_relations(
                    "Robot near sidewalk.", ["robot", "sidewalk"],
                    use_litellm_prompt=True, model="hosted_vllm/example",
                    response_hook=lambda *args: captured.append(args))
            self.assertEqual(captured, [('{"relations": []}', "length")])
