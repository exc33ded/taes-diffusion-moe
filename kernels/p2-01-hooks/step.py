# ---- Step 2.1 smoke test: 10 images, one direct model(x,t,y) call, domain=0, t=0.5 ----
tel = RouterTelemetry(model)
tel.register()
print("e_score_correction_bias all-zero per layer:",
      [bool((v == 0).all()) for v in tel.e_score_bias.values()], flush=True)

g = torch.Generator("cuda").manual_seed(0)
x = torch.randn(10, 4, 32, 32, device="cuda", generator=g)
y = torch.randint(0, 1000, (10,), device="cuda", generator=g)
t = torch.full((10,), 0.5, device="cuda")
tel.set_context(0, 0.5)
with torch.no_grad():
    out = model(x, t, y)
if isinstance(out, tuple):
    out = out[0]
print("output shape:", out.shape)
print("has NaNs:", bool(torch.isnan(out).any()))
print("freq summed over experts, per layer:", tel.freq[0, 25].sum(-1).tolist())
print("l2_count summed over experts, per layer:", tel.l2_count[0, 25].sum(-1).tolist())
print("t range seen:", tel.t_seen[0], tel.t_seen[1])
assert (tel.freq == tel.l2_count).all(), "router-side and expert-side counts differ"
print("freq == l2_count everywhere: True")
print("gate_sum mean per selected slot, layer 0:",
      float(tel.gate_sum[0, 25, 0].sum() / tel.freq[0, 25, 0].sum()))
os.makedirs(f"{RES}/telemetry", exist_ok=True)
tel.save(f"{RES}/telemetry/smoke_2_1.pt")
print("STEP 2.1 DONE", flush=True)
