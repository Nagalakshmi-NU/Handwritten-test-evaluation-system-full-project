/* ── AutoGrade Shared Profile + Theme JS ── */

// ── Dark/Light Mode ──────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('ag_theme') || 'dark';
  applyTheme(saved);
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('ag_theme', theme);
  const btn = document.getElementById('themeToggle');
  if (btn) btn.textContent = theme === 'dark' ? '☀️' : '🌙';
}

function toggleTheme() {
  const current = localStorage.getItem('ag_theme') || 'dark';
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

// Light mode overrides
const lightCSS = `
  [data-theme="light"] {
    --bg: #f0f4ff;
    --surface: rgba(255,255,255,.9);
    --surface2: rgba(240,244,255,.8);
    --border: rgba(0,0,0,.1);
    --border2: rgba(0,0,0,.15);
    --text: #0f172a;
    --muted: #64748b;
  }
  [data-theme="light"] body::before {
    background: radial-gradient(ellipse 60% 50% at 10% 10%, rgba(79,142,247,.06) 0%, transparent 60%),
                radial-gradient(ellipse 50% 60% at 90% 90%, rgba(124,58,237,.04) 0%, transparent 60%);
  }
  [data-theme="light"] nav {
    background: rgba(240,244,255,.9);
    border-bottom-color: rgba(0,0,0,.08);
  }
  [data-theme="light"] .grid-bg {
    background-image: linear-gradient(rgba(79,142,247,.06) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(79,142,247,.06) 1px, transparent 1px);
  }
  [data-theme="light"] .table-row:hover { background: rgba(79,142,247,.06); }
  [data-theme="light"] .profile-dropdown,
  [data-theme="light"] .notif-dropdown,
  [data-theme="light"] .profile-modal { background: rgba(240,244,255,.98); }
`;

// Inject light mode CSS
const styleEl = document.createElement('style');
styleEl.textContent = lightCSS;
document.head.appendChild(styleEl);

// ── Profile ──────────────────────────────────────────────────
async function loadProfile() {
  try {
    const res = await fetch('/me');
    if (!res.ok) return;
    const d = await res.json();
    const initials = d.initials || d.name.charAt(0).toUpperCase();
    const role = d.role;

    // Update avatar
    const av = document.getElementById('profileAvatar');
    if (av) { av.className = `avatar ${role}`; av.innerHTML = `<div class="avatar-pulse"></div>${initials}`; }

    // Update nav name
    const pn = document.getElementById('profileName');
    if (pn) pn.textContent = d.name.split(' ')[0];

    // Update dropdown
    const pdAv = document.getElementById('pdAvatar');
    if (pdAv) { pdAv.className = `pd-avatar ${role}`; pdAv.textContent = initials; }
    const pdName = document.getElementById('pdName');
    if (pdName) pdName.textContent = d.name;
    const pdEmail = document.getElementById('pdEmail');
    if (pdEmail) pdEmail.textContent = d.email;
    const pdId = document.getElementById('pdId');
    if (pdId) {
      const idLabel = role === 'teacher' ? 'Employee ID' : role === 'student' ? 'Student ID' : 'Admin ID';
      const idVal = d.employee_id || d.student_id || d.admin_id || '—';
      pdId.textContent = `${idLabel}: ${idVal}`;
    }

    // Store for edit modal
    window._profileData = d;
  } catch (e) {}
}

function toggleProfile() {
  const dd = document.getElementById('profileDropdown');
  if (dd) dd.classList.toggle('show');
}

// Close dropdown on outside click
document.addEventListener('click', e => {
  const pw = document.getElementById('profileWrap');
  if (pw && !pw.contains(e.target)) {
    const dd = document.getElementById('profileDropdown');
    if (dd) dd.classList.remove('show');
  }
});

// ── Profile Edit Modal ───────────────────────────────────────
function openEditProfile() {
  document.getElementById('profileDropdown')?.classList.remove('show');
  const d = window._profileData || {};
  document.getElementById('pmName').value = d.name || '';
  document.getElementById('pmOldPwd').value = '';
  document.getElementById('pmNewPwd').value = '';
  document.getElementById('pmAlert').className = 'pm-alert';
  document.getElementById('profileModalOverlay').classList.add('show');
}

function closeEditProfile() {
  document.getElementById('profileModalOverlay').classList.remove('show');
}

async function saveProfile() {
  const name    = document.getElementById('pmName').value.trim();
  const oldPwd  = document.getElementById('pmOldPwd').value;
  const newPwd  = document.getElementById('pmNewPwd').value;
  const alert   = document.getElementById('pmAlert');
  const btn     = document.getElementById('pmSaveBtn');

  if (!name) { showPmAlert('error', 'Name cannot be empty.'); return; }

  const body = { name };
  if (newPwd) { body.old_password = oldPwd; body.new_password = newPwd; }

  btn.disabled = true; btn.textContent = 'Saving...';
  try {
    const res  = await fetch('/me/update', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await res.json();
    if (res.ok) {
      showPmAlert('success', '✅ Profile updated successfully!');
      window._profileData = { ...window._profileData, name: data.name };
      loadProfile();
      setTimeout(closeEditProfile, 1500);
    } else {
      showPmAlert('error', data.error || 'Update failed.');
    }
  } catch (e) {
    showPmAlert('error', 'Network error.');
  }
  btn.disabled = false; btn.textContent = 'Save Changes';
}

function showPmAlert(type, msg) {
  const el = document.getElementById('pmAlert');
  el.textContent = msg;
  el.className = `pm-alert ${type} show`;
}

// ── Inject Profile Modal HTML ────────────────────────────────
function injectProfileModal() {
  if (document.getElementById('profileModalOverlay')) return;
  const html = `
    <div class="profile-modal-overlay" id="profileModalOverlay" onclick="if(event.target===this)closeEditProfile()">
      <div class="profile-modal">
        <button class="pm-close" onclick="closeEditProfile()">✕</button>
        <div class="pm-title">✏️ Edit Profile</div>
        <div class="pm-alert" id="pmAlert"></div>
        <div class="pm-field"><label class="pm-label">Full Name</label><input class="pm-input" id="pmName" type="text" placeholder="Your name"/></div>
        <div class="pm-field"><label class="pm-label">Current Password</label><input class="pm-input" id="pmOldPwd" type="password" placeholder="Enter current password to change it"/></div>
        <div class="pm-field"><label class="pm-label">New Password</label><input class="pm-input" id="pmNewPwd" type="password" placeholder="Leave blank to keep current"/></div>
        <button class="pm-btn" id="pmSaveBtn" onclick="saveProfile()">Save Changes</button>
      </div>
    </div>`;
  document.body.insertAdjacentHTML('beforeend', html);
}

// ── Inject Theme Toggle Button ───────────────────────────────
function injectThemeToggle() {
  if (document.getElementById('themeToggle')) return;
  const btn = document.createElement('button');
  btn.id = 'themeToggle';
  btn.onclick = toggleTheme;
  btn.style.cssText = 'position:fixed;bottom:24px;left:24px;z-index:500;width:42px;height:42px;border-radius:50%;border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.08);backdrop-filter:blur(10px);font-size:18px;cursor:pointer;transition:all .2s;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 16px rgba(0,0,0,.3)';
  btn.title = 'Toggle Dark/Light Mode';
  document.body.appendChild(btn);
}

// ── Inject Edit Profile link into dropdown ───────────────────
function injectEditProfileLink() {
  const pdBody = document.querySelector('.pd-body');
  if (!pdBody || pdBody.querySelector('.pd-edit')) return;
  const editBtn = document.createElement('button');
  editBtn.className = 'pd-item pd-edit';
  editBtn.onclick = openEditProfile;
  editBtn.innerHTML = '<span class="pd-icon">✏️</span>Edit Profile';
  const divider = pdBody.querySelector('.pd-divider');
  if (divider) pdBody.insertBefore(editBtn, divider);
  else pdBody.prepend(editBtn);
}

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  injectThemeToggle();
  injectProfileModal();
  loadProfile().then(() => injectEditProfileLink());
});
