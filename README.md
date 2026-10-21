Quran Cinematic AI Factory
مشروع Python 3.11 لإنتاج فيديو قرآني تعليمي عمودي من آية واحدة، ثم فحصه ونشره على YouTube وFacebook عند نجاح كل الفحوص. يتكون المستودع من خمسة ملفات مصدر فقط: main.py وservices.py وvideo_engine.py وrequirements.txt و.github/workflows/generate_video.yml.
تنبيه الحقوق قبل التشغيل
الوصول إلى API لا يمنح تلقائيًا حق إعادة نشر النصوص أو التلاوات أو التفاسير. شروط Quran.com تقيّد محتوى الخدمة للاستخدام الشخصي غير التجاري، وشروط مطوري Quran Foundation تسمح ببعض التطبيقات التجارية لكنها تمنع إعادة توزيع البيانات الخام أو إعادة ترخيصها، وتحيل إلى الترخيص الخاص بكل مصدر. لم يتضح من المصادر الرسمية التي راجعناها وجود إذن عام لإعادة نشر التلاوة أو «التفسير الميسر» تجاريًا. لذلك يوقف البرنامج الإنتاج ما لم يضف المشغّل معرّف المورد ورابط ترخيص HTTPS صريحًا إلى قائمتي السماح، ويؤكد حصوله على حق الاستخدام التجاري وإعادة النشر. لا تضف موردًا قبل الحصول على إذن يغطي الفيديو المنشور والمنصة والاستخدام التجاري عند انطباقه.
مصادر الشروط الرسمية: Quran.com Terms، Quran Foundation Developer Terms، Quran Foundation API docs، Pexels License. تأكيدات الترخيص في متغيرات البيئة هي إقرار من المشغّل وليست رأيًا قانونيًا أو تحققًا آليًا من صحة الترخيص.
بنية المشروع
text
Quran-Cinematic-AI-Factory/
├── .github/workflows/generate_video.yml
├── main.py
├── services.py
├── video_engine.py
├── requirements.txt
└── README.md
ينشئ البرنامج عند التشغيل ملفات بيانات فقط، مثل data/production_history.json وoutput/final_output.mp4 والتقارير. لا ينشئ ملفات مصدر إضافية.
ما يفعله النظام
يجلب النص العثماني من Quran Foundation Content API v4 ويتحقق من رقم السورة والآية. ثم يطلب مورد تلاوة محددًا ويقبل التوقيتات فقط عندما تعود فعليًا من الـAPI، وتطابق عدد كلمات الآية وترتيبها ومدة ملف الصوت. لا يسرّع التلاوة ولا يستبدلها بـTTS. يرفض الآية إذا لم توجد التوقيتات أو ترخيص المصدر.
يرسل البرنامج ملف التلاوة إلى Gemini في طلب منفصل من دون تمرير نص الآية المتوقع، ويفرّغ الكلمات المسموعة ثم يقارنها بعد إزالة التشكيل وعلامات الترقيم. أي كلمة ناقصة أو زائدة أو غير مؤكدة ترفض المرشح قبل الإنتاج. هذا فحص آلي وليس شهادة معصومة؛ يظل الاستماع البشري لعينة كل مصدر مهمًا، ويُحفظ التفريغ ونتيجة المطابقة في تقرير التشغيل.
يجلب التفسير من مورد محدد، وينظف HTML من دون إعادة كتابة المصدر، ويستخدم Gemini لتوليد خمس صيغ هوك وتقييمها، ثم يراجع النص الناتج بمقارنته بالمصدر. هذه المراجعة الآلية ليست مراجعة عالم شرعي ولا تضمن صحة التفسير؛ يرفض النظام النتائج التي تعيد needs_review أو لا تثبت ارتباطًا واضحًا بالمصدر.
ينشئ TTS لمقاطع التعليق فقط. عبارة «صدق الله العظيم» تُنشأ في ملف مستقل مرة واحدة بعد التلاوة، ويُمنع تكرارها في الحقول الأخرى. يطلب الفيديو صورة 1080×1920 و30fps وH.264/AAC، وخلفية طبيعية من Pexels، ويعرض الآية كاملة مع إبراز الكلمة اعتمادًا على توقيتات المصدر. إذا كانت مدة الصوت أقل من 45 ثانية، يمدد المشهد الأخير بصمت إلى الحد الأدنى بدل تسريع التلاوة أو قصها؛ وإذا كانت التلاوة تجعل المقطع أطول من 60 ثانية، يحتفظ بها كاملة.
يفحص البرنامج الملف باستخدام FFprobe وFFmpeg، بما في ذلك الأبعاد والترميزات والمعدل والمدة وتزامن مساري الصوت والصورة والشاشات السوداء الطويلة. يتم النشر فقط بعد التحقق. تحفظ حالة YouTube وFacebook ومعرّف كل منشور على حدة. لا تعاد محاولة رفع جديدة عند غموض نتيجة رفع أول؛ يُستخدم التحقق من المعرّف إذا صدر، لتجنب المنشورات المكررة.
التثبيت والتشغيل محليًا
يتطلب Python 3.11 وFFmpeg وFFprobe وخط Noto Naskh Arabic أو خطًا عربيًا مناسبًا. على Ubuntu/Debian:
Bash
sudo apt-get update
sudo apt-get install -y ffmpeg fontconfig fonts-noto-core
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
اضبط الأسرار والمتغيرات الموضحة أدناه في بيئتك بطريقة آمنة، ثم شغّل:
Bash
python main.py --mode auto
python main.py --mode manual --chapter 2 --verse 152
لا تمرر المفاتيح في سطر الأوامر أو تضعها في المستودع. في الوضع اليدوي لا يستبدل النظام الآية المطلوبة بآية أخرى إذا فشل التحقق.
شغّل اختبارات الوحدة المحلية التي لا تتصل بالخدمات الخارجية:
Bash
python services.py
python video_engine.py
python main.py --test
هذه الاختبارات لا تمثل اختبارًا حيًا لـQuran Foundation أو Gemini أو Pexels أو YouTube أو Meta، ولا تنشر فيديو.
إعداد GitHub Actions
ارفع الملفات الخمسة إلى مستودع GitHub خاص أو عام، وتأكد أن ملف الـworkflow موجود على الفرع الافتراضي. التشغيل اليدوي على فرع آخر يتخطاه الـworkflow كي يبقى سجل منع التكرار موحدًا.
أضف المفاتيح السرية التالية من Settings → Secrets and variables → Actions → Secrets:
GEMINI_API_KEY
QF_CLIENT_ID وQF_CLIENT_SECRET لـQuran Foundation؛ يطلب البرنامج access token عبر client credentials ويجدده قبل انتهاء الصلاحية. لا تضف QF_AUTH_TOKEN ثابتًا.
PEXELS_API_KEY
YOUTUBE_CLIENT_ID وYOUTUBE_CLIENT_SECRET وYOUTUBE_REFRESH_TOKEN؛ يجب إصدار الرمز مجددًا بنطاقي OAuth youtube.upload وyoutube.readonly لأن النظام يرفع الفيديو ثم يقرأ حالة المعالجة.
META_PAGE_ACCESS_TOKEN وMETA_PAGE_ID لصفحة Facebook وصلاحيات Reels المطلوبة
أضف المتغيرات غير السرية في Variables، بعد التحقق من الحقوق:
GEMINI_MODEL (الافتراضي gemini-3.8-flash)
QURAN_TEXT_LICENSE_URL وQURAN_TEXT_LICENSE_CONFIRMED=true بعد التحقق من حق إعادة نشر النص القرآني من المصدر المحدد.
RECITATION_ID وRECITATION_LICENSE_URL وRECITATION_LICENSE_CONFIRMED=true
RECITATION_ALLOWLIST بصيغة معرّف|رابط-ترخيص-HTTPS، ويمكن إضافة أكثر من مورد بفصلها بفاصلة منقوطة ;
RECITATION_REQUIRE_MISHARY=true افتراضيًا. لا تستخدم قارئًا آخر إلا بعد ضبط مورد مرخص والتحقق منه، ثم ضع القيمة false إذا كان ذلك مناسبًا.
TAFSIR_ID وTAFSIR_LICENSE_URL وTAFSIR_COMMERCIAL_LICENSE_VERIFIED=true
TAFSIR_ALLOWLIST بالصيغة نفسها معرّف|رابط-ترخيص-HTTPS؛ القائمة فارغة افتراضيًا ويرفض البرنامج أي تفسير غير موجود فيها.
PEXELS_LICENSE_CONFIRMED=true بعد مراجعة الترخيص الحالي والقيود المتعلقة بالأشخاص والعلامات والحقوق الأخرى.
TTS_TERMS_CONFIRMED=true وTTS_LICENSE_URL بعد التحقق بنفسك من سماح الشروط السارية بالاستخدام التجاري وإعادة نشر الصوت المولّد. إذا لم تستطع التحقق، لا تفعّل الإقرار.
RECITATION_ATTRIBUTION وTAFSIR_ATTRIBUTION وQURAN_ATTRIBUTION وTTS_ATTRIBUTION عند اشتراط الإسناد في الترخيص.
لا تضبط ARABIC_FONT عادةً؛ يستخدم المشروع خطوط Noto ذات ترخيص SIL OFL 1.1. عند اختيار خط مخصص، أضف ARABIC_FONT_LICENSE_URL وARABIC_FONT_LICENSE_CONFIRMED=true بعد التحقق.
META_REELS_DISCLOSURE_CONFIRMED=true فقط بعد تجهيز متطلبات الإفصاح وخيارات التحكم التي يطلبها دليل Meta Reels. النشر على Meta معطّل افتراضيًا؛ اضبط META_PUBLISH_ENABLED=true بعد اختبار التطبيق والصلاحيات.
TTS_VOICE (الافتراضي ar-EG-ShakerNeural)، وMETA_GRAPH_VERSION (الافتراضي v26.0)، وYOUTUBE_PRIVACY_STATUS (الافتراضي private). اجعل الفيديو public فقط بعد مراجعة مخرجات الاختبار يدويًا.
قد يفرض Google OAuth تهيئة شاشة موافقة أو إدراج مستخدم اختبار أو التحقق من التطبيق. يجب أن يكون Refresh Token صالحًا ومخوّلًا على القناة المراد النشر فيها.
يحتاج الـworkflow إلى contents: write كي يحفظ data/production_history.json بين التشغيلات؛ النشر إلى المنصات يحتاج الأسرار والصلاحيات المستقلة أعلاه. يرفع الفيديو والتقرير كـArtifact لمدة 30 يومًا.
الـworkflow مجدول يوميًا في Africa/Casablanca عند الدقيقة 17 لتجنب رأس الساعة، مع تشغيل يدوي يتيح auto أو manual ورقم السورة والآية. قد يؤخر GitHub التشغيل المجدول. ينطبق حد الفيديو اليومي على الوضع التلقائي فقط؛ ولا يمنع التشغيل اليدوي إذا سبق نشر فيديو تلقائي في التاريخ نفسه. يمكن إعادة محاولة فشل معروف قبل النشر، بينما تمنع الحالات غير المحسومة رفعًا جديدًا إلى أن تُستعد.
متغيرات الإعداد
القيم التالية تقرأ من البيئة وتُقص المسافات الزائدة: TZ (الافتراضي Africa/Casablanca)، QF_ENV (production أو prelive، والافتراضي production؛ يجب أن يطابق مضيف OAuth ومضيف Content API)، DAILY_VIDEO_COUNT=1، VIDEO_WIDTH=1080، VIDEO_HEIGHT=1920، VIDEO_FPS=30، TARGET_MIN_DURATION=45، TARGET_MAX_DURATION=60، MAX_CANDIDATES_PER_RUN=5، MAX_RETRIES=3، TTS_VOICE=ar-EG-ShakerNeural، OUTPUT_DIR=output، HISTORY_FILE=data/production_history.json، وFINAL_VIDEO_NAME=final_output.mp4.
في الوضع التلقائي، ينتقي النظام من مجموعة آيات محدودة ومضمّنة في main.py، ويستبعد الآيات المنشورة. إذا فشلت المرشحات الخمسة، ينتهي التشغيل دون نشر. لا تعني قائمة السماح أن معنى الآية أو ترخيصها قد تم التحقق شرعيًا أو قانونيًا على نحو مستقل.
السجل والاستعادة
يكتب السجل ذريًا إلى data/production_history.json مع قفل ملفات محلي. يحتفظ بالمصادر والتراخيص المُدخلة، ونتائج الفحوص، وبصمة SHA-256، وعدد المحاولات، وحالة كل منصة. يعتمد منع التكرار عبر تشغيلات GitHub على إنشاء commit للسجل ثم إعادة تأسيسه على أحدث نسخة قبل الدفع. إذا تعذر rebase، يفشل حفظ السجل بدل الكتابة فوق بيانات أحدث؛ راجع سجل Actions ثم عالج التعارض يدويًا.
لاستعادة نشر منصة لم تؤكد نجاحها، نزّل نسخة Artifact ذاتها واستخدم --publish-only مع الآية والملف. يتحقق الأمر من بصمة الملف المسجلة ويتخطى أي منصة لديها نشر مؤكد:
Bash
python main.py --publish-only --chapter 2 --verse 152 --input-video /path/to/final_output.mp4
إذا لم تتوفر نسخة Artifact مطابقة، يرفض البرنامج إعادة رفع فيديو مختلف ضمن استعادة السجل. لا تستخدم --reprocess كطريقة لإعادة نشر؛ هو يسمح بإعادة إنتاج الملف فقط مع تخطي الوجهات المنشورة سابقًا.
التوثيق الرسمي للتكاملات
Quran Foundation Content API v4: verses by key، recitations، audio by chapter، audio file URLs and CDN، tafsir by ayah.
Gemini: Structured outputs، Audio understanding، وPython SDK.
Pexels: API documentation وLicense.
YouTube: videos.insert، OAuth authentication، videos.list.
Meta: Reels Publishing API.
GitHub Actions: schedule events وworkflow syntax.
Arabic font licensing: SIL Open Font License.
استكشاف الأخطاء
رفض الحقوق: تحقق من رابط HTTPS، ومعرّف المورد، وإدراج الزوج الدقيق في قائمة السماح، ثم احتفظ بدليل الترخيص خارج المستودع. لا تعطل الحاجز لمجرد أن الخدمة تسمح بالتنزيل.
لا توجد توقيتات كلمات: يرفض النظام الآية؛ لا يوزع مدة الصوت بالتساوي ولا يخمّن. تحقق من رقم القارئ وتوفر segments في استجابة Quran Foundation.
رفض التلاوة: يفرض الإعداد الافتراضي أن يطابق المورد مشاري العفاسي. تحقق من معرّف RECITATION_ID في كتالوج API أو اضبط قارئًا بديلًا مرخصًا بشكل صريح.
Gemini رفض المخطط أو المحتوى: تحقق من صلاحية المفتاح والنموذج المتاح، ثم راجع تقرير العملية. لا تغيّر النص القرآني لإرضاء النموذج.
خط عربي غير ظاهر: ثبّت Noto Naskh Arabic أو اضبط ARABIC_FONT إلى ملف خط عربي مرخص محليًا، مع ARABIC_FONT_LICENSE_URL وARABIC_FONT_LICENSE_CONFIRMED=true.
فشل FFmpeg: افحص ffmpeg -version وffprobe -version وfc-match 'Noto Naskh Arabic'، ثم نزّل Artifact التشخيص.
YouTube أو Meta غير مؤكد: يعتمد النجاح على استجابة المعالجة والظهور النهائي، لا على رفع الملف فقط. لا تعاود الرفع مباشرة بعد انقطاع غير محسوم؛ احتفظ بمعرّف المنصة وافحص حالته أولًا.
عدم وجود نشر: إذا كانت بيانات اعتماد منصة مفقودة أو صلاحياتها غير كافية، تسجل المنصة كفاشلة ويخرج التشغيل برمز غير صفري. لن يسجل النظام نجاحًا وهميًا.
