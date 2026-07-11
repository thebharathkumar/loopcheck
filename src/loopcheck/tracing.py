import json
import sqlite3
from collections.abc import Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class SqliteSpanExporter(SpanExporter):
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            ctx = span.get_span_context()
            attrs = dict(span.attributes or {})
            parent_id = f"{span.parent.span_id:016x}" if span.parent else None
            self._conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"{ctx.span_id:016x}",
                    f"{ctx.trace_id:032x}",
                    parent_id,
                    span.name,
                    span.start_time,
                    span.end_time,
                    json.dumps(attrs, default=str),
                    attrs.get("loopcheck.run_id"),
                ),
            )
        self._conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def init_tracing(
    conn: sqlite3.Connection, otlp_endpoint: str | None = None
) -> trace.Tracer:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(SqliteSpanExporter(conn)))
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    return provider.get_tracer("loopcheck")
