package dev.portableapp.shell;

import static org.junit.Assert.*;

import android.webkit.WebView;
import androidx.test.core.app.ActivityScenario;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.io.FileInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/** Queries the APK's real index through the same form and fetch path users use. */
@RunWith(AndroidJUnit4.class)
public final class RagOfflineTest {
    private static String evaluate(ActivityScenario<MainActivity> scenario, String script) throws Exception {
        AtomicReference<String> result = new AtomicReference<>();
        CountDownLatch done = new CountDownLatch(1);
        scenario.onActivity(activity -> ((WebView) activity.findViewById(MainActivity.WEB_VIEW_ID))
                .evaluateJavascript(script, value -> { result.set(value); done.countDown(); }));
        assertTrue("WebView did not evaluate JavaScript", done.await(10, TimeUnit.SECONDS));
        return result.get();
    }

    private static void awaitTrue(ActivityScenario<MainActivity> scenario, String script) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(40);
        while (System.nanoTime() < deadline) {
            if ("true".equals(evaluate(scenario, script))) return;
            Thread.sleep(100);
        }
        fail("Real application did not satisfy: " + script);
    }

    private static PortableRuntime.Session start(ActivityScenario<MainActivity> scenario) throws Exception {
        PortableRuntime.Session session = PortableRuntime.obtain(ApplicationProvider.getApplicationContext());
        awaitTrue(scenario, "document.readyState === 'complete' && document.querySelector('#ask-form') !== null");
        // Observe a cloned real response. Never replace a request or return a fixture.
        evaluate(scenario, "window.__ragReply=null;window.__ragFetch=window.fetch.bind(window);"
                + "window.fetch=async function(...args){const r=await window.__ragFetch(...args);"
                + "if(args[0]==='/v1/ask'){window.__ragReply={status:r.status,body:await r.clone().json()};}return r;};true");
        return session;
    }

    private static JSONObject submit(ActivityScenario<MainActivity> scenario, String profile, String question) throws Exception {
        evaluate(scenario, "document.querySelector('#profile').value=" + JSONObject.quote(profile) + ";"
                + "document.querySelector('#profile').dispatchEvent(new Event('change'));"
                + "document.querySelector('#question').value=" + JSONObject.quote(question) + ";"
                + "document.querySelector('#question').dispatchEvent(new Event('input'));"
                + "window.__ragReply=null;document.querySelector('#ask-form').requestSubmit();true");
        awaitTrue(scenario, "window.__ragReply !== null && document.querySelector('#result').getAttribute('aria-busy') === 'false'");
        String encoded = evaluate(scenario, "JSON.stringify(window.__ragReply)");
        JSONObject reply = new JSONObject((String) new JSONTokener(encoded).nextValue());
        assertEquals(reply.toString(), 200, reply.getInt("status"));
        assertTrue("Submitted question was not shown in the page", Boolean.parseBoolean(evaluate(scenario,
                "document.querySelector('#submitted-question').textContent.includes(" + JSONObject.quote(question) + ")")));
        return reply.getJSONObject("body");
    }

    private static JSONObject bundledIndex(PortableRuntime.Session session) throws Exception {
        File index = new File(session.resourceDirectory, "data/portable-index.json");
        try (FileInputStream input = new FileInputStream(index)) {
            ByteArrayOutputStream data = new ByteArrayOutputStream();
            BoundaryPolicy.copyBounded(input, data, 16L * 1024 * 1024);
            return new JSONObject(data.toString(StandardCharsets.UTF_8.name()));
        }
    }

    @Test public void bothProfilesReturnOriginalCorpusEvidenceThroughTheRealForm() throws Exception {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            PortableRuntime.Session session = start(scenario);
            JSONObject index = bundledIndex(session);
            String[][] cases = {
                {"en-v1", "FAISS cosine similarity normalized vectors on my Android device", "faiss-readme:0002"},
                {"zh-v1", "BGE-M3 稀疏检索和多向量有什么区别？在我的手机上检索", "flagembedding-readme-zh:0021"}
            };
            for (String[] query : cases) {
                JSONObject reply = submit(scenario, query[0], query[1]);
                assertEquals("answer", reply.getString("status"));
                assertEquals(query[0], reply.getString("profile"));
                assertEquals("portable-index-v1", reply.getString("runtime"));
                assertEquals(session.metadata.getString("build_revision"), reply.getString("index_revision"));
                assertEquals(index.getString("revision"), reply.getString("index_revision"));
                assertTrue(reply.getString("index_revision").matches("[0-9a-f]{40}"));
                JSONArray evidence = reply.getJSONArray("evidence");
                assertEquals(3, evidence.length());
                assertEquals(query[2], evidence.getJSONObject(0).getString("chunk_id"));
                JSONArray chunks = index.getJSONObject("profiles").getJSONObject(query[0]).getJSONArray("chunks");
                for (int ordinal = 0; ordinal < evidence.length(); ordinal++) {
                    JSONObject item = evidence.getJSONObject(ordinal);
                    JSONObject original = null;
                    for (int i = 0; i < chunks.length(); i++) {
                        JSONObject candidate = chunks.getJSONObject(i);
                        if (candidate.getString("chunk_id").equals(item.getString("chunk_id"))) original = candidate;
                    }
                    assertNotNull("Returned evidence does not exist in the packaged corpus", original);
                    assertEquals(original.getString("text"), item.getString("text"));
                    assertEquals(original.getString("source_url"), item.getString("source_url"));
                    assertTrue(item.getDouble("score") > 0);
                    String displayed = evaluate(scenario, "document.querySelectorAll('#evidence .source-details pre')["
                            + ordinal + "].textContent");
                    assertEquals(item.getString("text"), new JSONTokener(displayed).nextValue());
                }
                String firstId = evidence.getJSONObject(0).getString("chunk_id");
                assertEquals(firstId, reply.getJSONArray("citations").getJSONObject(0).getString("chunk_id"));
                assertTrue(reply.getString("answer").contains(evidence.getJSONObject(0).getString("text")));
            }
        }
    }

    @Test public void unseenOutOfCorpusQuestionAbstainsWithoutInventingEvidence() throws Exception {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            start(scenario);
            JSONObject reply = submit(scenario, "en-v1", "unknownterm" + UUID.randomUUID().toString().replace("-", ""));
            assertEquals("abstain", reply.getString("status"));
            assertEquals("insufficient_evidence", reply.getString("reason"));
            assertEquals(0, reply.getJSONArray("evidence").length());
            assertEquals(0, reply.getJSONArray("citations").length());
            assertEquals("0", evaluate(scenario, "document.querySelectorAll('#evidence .evidence-card').length"));
        }
    }

    @Test public void mobileCapabilitiesDoNotPretendToRunTheDesktopBenchmark() throws Exception {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            start(scenario);
            evaluate(scenario, "window.__ragLimits=null;Promise.all([fetch('/health').then(r=>r.json()),"
                    + "fetch('/v1/evaluations',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(r=>r.status)])"
                    + ".then(r=>window.__ragLimits={health:r[0],evaluationStatus:r[1]});true");
            awaitTrue(scenario, "window.__ragLimits !== null");
            JSONObject limits = new JSONObject((String) new JSONTokener(
                    evaluate(scenario, "JSON.stringify(window.__ragLimits)")).nextValue());
            JSONObject capabilities = limits.getJSONObject("health").getJSONObject("capabilities");
            assertTrue(capabilities.getBoolean("offline_search"));
            assertFalse(capabilities.getBoolean("semantic_rerank"));
            assertFalse(capabilities.getBoolean("benchmark_cli"));
            assertEquals(501, limits.getInt("evaluationStatus"));
        }
    }
}
