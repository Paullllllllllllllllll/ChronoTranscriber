"""Batch transcription on the shared backends of OpenAI, Anthropic and Google.

:mod:`~chronotranscriber.batch.bodies` builds the request bodies,
:mod:`~chronotranscriber.batch.submit` submits a document's pages as store
submissions, :mod:`~chronotranscriber.batch.finalize` finalizes a finished
batch into the store for batch check (:mod:`~chronotranscriber.batch.check`)
and repair (:mod:`~chronotranscriber.batch.repair`), and
:mod:`~chronotranscriber.batch.cancel` cancels and lists batches; the shared
:mod:`~chronotranscriber.common.batch_reconcile` settles unresolved
submissions. Every API call goes through
:mod:`~chronotranscriber.batch.clients`.
"""
