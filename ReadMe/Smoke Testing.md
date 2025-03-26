# ✅ Smoke Testing Error Criteria Checklist

Smoke testing (also known as "sanity testing") is a shallow and wide approach to testing that validates the critical functionality of a system after a new build or deployment. Its goal is to catch showstoppers early and determine whether the build is stable enough for further testing.

---

## 1. Core Functionality Verification
- [ ] Confirm the system starts without crashes or critical errors.
- [ ] Validate that essential services, microservices, or background processes are running.
- [ ] Navigate through key pages or screens (e.g., homepage, login, dashboard).
- [ ] Test navigation elements (menus, tabs, back/forward buttons, links).
- [ ] Input basic data into a form and verify that it is saved and retrieved correctly.
- [ ] Perform a basic transaction or workflow (e.g., submit a form, load a report).
- [ ] Confirm that core features (e.g., login, search, checkout) are working.

---

## 2. Key Component Validation
- [ ] Verify database connectivity and ability to run basic queries (insert, update, select).
- [ ] Confirm APIs return valid responses for basic requests.
- [ ] Validate basic endpoint availability and proper status codes (200 OK, 401 Unauthorized).
- [ ] Test internal or external service calls for availability and responsiveness.
- [ ] Verify system connectivity to essential network resources or integrations.
- [ ] Check system clock/time sync (if applicable) for time-sensitive features.

---

## 3. Major Error Detection
- [ ] Check for application crashes immediately after launch.
- [ ] Detect any critical system errors or exception screens.
- [ ] Identify blocking issues (e.g., login fails, dashboard doesn’t load).
- [ ] Ensure that the UI renders correctly and main components are visible.
- [ ] Look for missing or broken assets (CSS, JS, images) that could hinder usability.

---

## 🔄 Key Characteristics of Smoke Testing
- [ ] **Quick Execution**: Meant to complete in minutes, not hours.
- [ ] **Surface-Level Coverage**: Focuses on breadth over depth.
- [ ] **Early Indicator**: Conducted immediately after a new build is deployed.
- [ ] **Build Validation**: Confirms if a build is testable before deeper testing.
- [ ] **Go/No-Go Decision**: Used to decide whether to proceed with further testing.
- [ ] **Automation-Friendly**: Automation is highly recommended for repeatability and speed.

---

## 🔍 Differences from Regression Testing
- **Scope**:  
  - Smoke Testing: Narrow — only verifies critical functions.
  - Regression Testing: Broad — tests everything that could be affected by changes.
- **Depth**:  
  - Smoke Testing: Superficial.
  - Regression Testing: In-depth, covering edge cases and dependencies.
- **Timing**:  
  - Smoke Testing: Performed after each new build.
  - Regression Testing: Performed after changes or fixes are introduced.
- **Purpose**:  
  - Smoke Testing: Confirms basic operability.
  - Regression Testing: Ensures no functionality is broken due to updates.

