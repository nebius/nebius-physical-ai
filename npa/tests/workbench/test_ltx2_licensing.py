"""The LTX-2.5 licensing gate is a legal mechanism, so it is tested as one.

Two properties matter here and neither is provable by reading the module:

1. Nothing proceeds without an explicit, complete operator declaration. Every
   missing or malformed answer must refuse, because a default would mean Nebius
   answered a licensing question on the operator's behalf.
2. Attachment A(18) is enforced fail-closed downstream. Unlabelled artifacts,
   unknown schemas, and unknown dispositions must all deny — the permissive
   branch has to be the narrow, explicit one.
"""

from __future__ import annotations

import pytest

from npa.workbench.ltx2 import licensing


def env(**overrides: str) -> dict[str, str]:
    """Return a complete, valid declaration with *overrides* applied."""

    base = {
        licensing.ACCEPT_ENV: "YES",
        licensing.ENTITY_CLASS_ENV: licensing.ENTITY_COMMUNITY,
        licensing.USE_CLASS_ENV: licensing.USE_NON_COMMERCIAL,
    }
    base.update(overrides)
    return {key: value for key, value in base.items() if value is not None}


class TestDeclarationRefusals:
    def test_accepts_a_complete_declaration(self) -> None:
        declaration = licensing.declaration_from_env(env())
        assert declaration.entity_class == licensing.ENTITY_COMMUNITY
        assert declaration.use_class == licensing.USE_NON_COMMERCIAL

    def test_refuses_without_licence_acceptance(self) -> None:
        with pytest.raises(licensing.LtxLicenseError) as excinfo:
            licensing.declaration_from_env(env(**{licensing.ACCEPT_ENV: ""}))
        assert "Nothing has been downloaded" in str(excinfo.value)

    @pytest.mark.parametrize("value", ["", "y", "true", "1", "no", "YES please"])
    def test_only_an_exact_yes_accepts(self, value: str) -> None:
        with pytest.raises(licensing.LtxLicenseError):
            licensing.declaration_from_env(env(**{licensing.ACCEPT_ENV: value}))

    def test_acceptance_is_case_insensitive(self) -> None:
        assert licensing.declaration_from_env(env(**{licensing.ACCEPT_ENV: "yes"}))

    @pytest.mark.parametrize("value", ["", "startup", "small", "enterprise", "unknown"])
    def test_refuses_an_unrecognised_entity_class(self, value: str) -> None:
        with pytest.raises(licensing.LtxLicenseError) as excinfo:
            licensing.declaration_from_env(env(**{licensing.ENTITY_CLASS_ENV: value}))
        assert licensing.ENTITY_CLASS_ENV in str(excinfo.value)

    @pytest.mark.parametrize("value", ["", "research", "internal", "prod"])
    def test_refuses_an_unrecognised_use_class(self, value: str) -> None:
        with pytest.raises(licensing.LtxLicenseError) as excinfo:
            licensing.declaration_from_env(env(**{licensing.USE_CLASS_ENV: value}))
        assert licensing.USE_CLASS_ENV in str(excinfo.value)

    def test_refusal_links_the_terms_being_accepted(self) -> None:
        """An operator cannot accept terms the refusal does not name."""

        with pytest.raises(licensing.LtxLicenseError) as excinfo:
            licensing.declaration_from_env({})
        message = str(excinfo.value)
        assert licensing.LICENSE_URL in message
        assert licensing.ACCEPTABLE_USE_POLICY_URL in message
        assert licensing.LICENSE_DATE in message
        assert licensing.WEIGHTS_REPO_URL in message
        assert licensing.SOURCE_REF in message

    def test_refusal_warns_that_commercial_use_blocks_policy_training(self) -> None:
        """The Attachment A(18) trap is the one an operator is most likely to hit."""

        message = str(licensing.refusal_text("test"))
        assert "Attachment A(18)" in message
        assert "Robot policies are other machine learning models" in message


class TestSection21PaidLicence:
    def test_commercial_entity_commercial_use_needs_an_agreement_reference(
        self,
    ) -> None:
        with pytest.raises(licensing.LtxLicenseError) as excinfo:
            licensing.declaration_from_env(
                env(
                    **{
                        licensing.ENTITY_CLASS_ENV: licensing.ENTITY_COMMERCIAL,
                        licensing.USE_CLASS_ENV: licensing.USE_COMMERCIAL,
                    }
                )
            )
        assert licensing.COMMERCIAL_LICENSE_CONTACT in str(excinfo.value)

    def test_commercial_entity_commercial_use_proceeds_with_an_agreement(self) -> None:
        declaration = licensing.declaration_from_env(
            env(
                **{
                    licensing.ENTITY_CLASS_ENV: licensing.ENTITY_COMMERCIAL,
                    licensing.USE_CLASS_ENV: licensing.USE_COMMERCIAL,
                    licensing.COMMERCIAL_AGREEMENT_ENV: "CUA-2026-0042",
                }
            )
        )
        assert declaration.requires_paid_license
        assert declaration.commercial_agreement_ref == "CUA-2026-0042"

    def test_whitespace_is_not_an_agreement_reference(self) -> None:
        with pytest.raises(licensing.LtxLicenseError):
            licensing.declaration_from_env(
                env(
                    **{
                        licensing.ENTITY_CLASS_ENV: licensing.ENTITY_COMMERCIAL,
                        licensing.USE_CLASS_ENV: licensing.USE_COMMERCIAL,
                        licensing.COMMERCIAL_AGREEMENT_ENV: "   ",
                    }
                )
            )

    def test_commercial_entity_may_evaluate_under_the_section_2_2_carve_out(
        self,
    ) -> None:
        """A dev-VM evaluation by a large company is the Section 2.2 case."""

        declaration = licensing.declaration_from_env(
            env(**{licensing.ENTITY_CLASS_ENV: licensing.ENTITY_COMMERCIAL})
        )
        assert not declaration.requires_paid_license
        assert declaration.relies_on_non_commercial_carve_out

    def test_small_entity_commercial_use_needs_no_paid_licence(self) -> None:
        declaration = licensing.declaration_from_env(
            env(**{licensing.USE_CLASS_ENV: licensing.USE_COMMERCIAL})
        )
        assert not declaration.requires_paid_license


class TestAttachmentA18Disposition:
    def test_commercial_use_prohibits_training_other_models(self) -> None:
        declaration = licensing.declaration_from_env(
            env(**{licensing.USE_CLASS_ENV: licensing.USE_COMMERCIAL})
        )
        assert declaration.derived_model_training == licensing.TRAINING_PROHIBITED

    def test_non_commercial_use_permits_training_non_commercially(self) -> None:
        declaration = licensing.declaration_from_env(env())
        assert (
            declaration.derived_model_training
            == licensing.TRAINING_NON_COMMERCIAL_ONLY
        )

    def test_the_use_decides_not_the_entity_size(self) -> None:
        """Attachment A(18) is scoped by use, so a small company is not exempt."""

        small_commercial = licensing.declaration_from_env(
            env(
                **{
                    licensing.ENTITY_CLASS_ENV: licensing.ENTITY_COMMUNITY,
                    licensing.USE_CLASS_ENV: licensing.USE_COMMERCIAL,
                }
            )
        )
        assert small_commercial.derived_model_training == licensing.TRAINING_PROHIBITED


class TestProvenanceRecord:
    def test_records_the_terms_that_travel_with_the_output(self) -> None:
        record = licensing.ProvenanceRecord(
            declaration=licensing.declaration_from_env(env()),
            run_id="run-1",
            outputs=("s3://bucket/run-1/clip.mp4",),
            model_files=("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",),
        )
        payload = record.as_dict()

        assert payload["schema"] == licensing.PROVENANCE_SCHEMA
        assert payload["license"]["osi_approved"] is False
        assert payload["license"]["date"] == licensing.LICENSE_DATE
        assert payload["weights"]["delivery"] == "runtime-fetch-under-operator-hf-token"
        assert payload["restrictions"]["synthetic_content_disclosure"] == "required"
        assert payload["outputs"][0]["machine_generated"] is True

    def test_carries_the_output_obligations_that_survive_into_artifacts(self) -> None:
        record = licensing.ProvenanceRecord(
            declaration=licensing.declaration_from_env(env()), run_id="run-1"
        )
        obligations = record.as_dict()["restrictions"]["output_obligations"]
        assert "attachment-a-5-disclose-machine-generated" in obligations
        assert "attachment-a-19-no-stripping-provenance-or-watermarks" in obligations


class TestTrainingGateFailsClosed:
    def test_allows_a_non_commercial_run(self) -> None:
        record = licensing.ProvenanceRecord(
            declaration=licensing.declaration_from_env(env()), run_id="run-1"
        )
        decision = licensing.check_training_consumer(
            record.as_dict(), consumer="lerobot-policy"
        )
        assert decision.allowed
        assert "Derivative of LTX-2.x" in decision.reason

    def test_denies_a_commercial_run(self) -> None:
        record = licensing.ProvenanceRecord(
            declaration=licensing.declaration_from_env(
                env(**{licensing.USE_CLASS_ENV: licensing.USE_COMMERCIAL})
            ),
            run_id="run-1",
        )
        decision = licensing.check_training_consumer(
            record.as_dict(), consumer="lerobot-policy"
        )
        assert not decision.allowed
        assert "Attachment A(18)" in decision.reason

    @pytest.mark.parametrize("manifest", [None, "", [], 0, "{}"])
    def test_denies_when_no_manifest_accompanies_the_artifacts(
        self, manifest: object
    ) -> None:
        decision = licensing.check_training_consumer(manifest, consumer="trainer")
        assert not decision.allowed
        assert "provenance manifest" in decision.reason

    def test_denies_an_unrecognised_schema(self) -> None:
        decision = licensing.check_training_consumer(
            {"schema": "npa.ltx2.provenance.v99", "restrictions": {}},
            consumer="trainer",
        )
        assert not decision.allowed

    @pytest.mark.parametrize(
        "restrictions",
        [
            {},
            {"derived_model_training": ""},
            {"derived_model_training": "permitted"},
            {"derived_model_training": "allowed"},
        ],
    )
    def test_denies_an_unknown_disposition(self, restrictions: dict) -> None:
        decision = licensing.check_training_consumer(
            {"schema": licensing.PROVENANCE_SCHEMA, "restrictions": restrictions},
            consumer="trainer",
        )
        assert not decision.allowed

    def test_a_stripped_restrictions_block_does_not_open_the_gate(self) -> None:
        """Removing the restriction must not read as absence of restriction."""

        record = licensing.ProvenanceRecord(
            declaration=licensing.declaration_from_env(
                env(**{licensing.USE_CLASS_ENV: licensing.USE_COMMERCIAL})
            ),
            run_id="run-1",
        )
        payload = record.as_dict()
        payload.pop("restrictions")
        assert not licensing.check_training_consumer(payload, consumer="t").allowed
