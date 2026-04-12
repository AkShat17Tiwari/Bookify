import type { Variants } from 'framer-motion';

export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 30 },
  visible: (i: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.08,
      duration: 0.6,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  }),
};

export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { duration: 0.5, ease: 'easeOut' },
  },
};

export const slideInLeft: Variants = {
  hidden: { opacity: 0, x: -40 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] },
  },
};

export const slideInRight: Variants = {
  hidden: { opacity: 0, x: 40 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] },
  },
};

export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.92 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: { duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] },
  },
};

export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.06,
      delayChildren: 0.1,
    },
  },
};

export const staggerContainerSlow: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.12,
      delayChildren: 0.2,
    },
  },
};

export const floatAnimation = {
  y: [0, -12, -6, 0],
  rotate: [0, 1, -0.5, 0],
  transition: {
    duration: 6,
    repeat: Infinity,
    ease: 'easeInOut',
  },
};

export const glowPulse = {
  boxShadow: [
    '0 0 20px rgba(74, 144, 217, 0.2), 0 0 40px rgba(139, 92, 246, 0.1)',
    '0 0 30px rgba(74, 144, 217, 0.35), 0 0 60px rgba(139, 92, 246, 0.2)',
    '0 0 20px rgba(74, 144, 217, 0.2), 0 0 40px rgba(139, 92, 246, 0.1)',
  ],
  transition: {
    duration: 3,
    repeat: Infinity,
    ease: 'easeInOut',
  },
};

export const heartPop = {
  scale: [1, 1.3, 0.95, 1.15, 1],
  transition: {
    duration: 0.5,
    ease: 'easeInOut',
  },
};

export const pressIn = {
  scale: 0.97,
  transition: { duration: 0.1 },
};

export const hoverLift = {
  y: -8,
  transition: { duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] },
};

export const pageTransition: Variants = {
  initial: { opacity: 0, y: 20 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] },
  },
  exit: {
    opacity: 0,
    y: -10,
    transition: { duration: 0.3, ease: 'easeIn' },
  },
};

export const cardHover3D = (rotateX: number, rotateY: number) => ({
  rotateX,
  rotateY,
  transition: { duration: 0.1, ease: 'linear' },
});
